# scripts/inference_multihead.py
"""
Robust inference helper with verbose debug mode.

Usage:
  python3 scripts/inference_multihead.py --model models/cnn_multi.pth --h5 ml_dataset_v2.h5 --idx 0
  python3 scripts/inference_multihead.py --model models/cnn_multi.pth --h5 ml_dataset_v2.h5 --idx 0 --debug

What this version adds:
 - --debug prints checkpoint keys, meta, state_dict keys/shapes
 - Dumps HDF5 keys + shapes + sample dtypes
 - Tries to auto-fix common state_dict 'module.' prefix issue
 - Attempts to infer model's expected input channels from first conv weight
 - Detailed traceback and helpful suggestions when an error occurs
"""
import argparse
import ast
import h5py
import numpy as np
import os
import sys
import traceback
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def safe_eval_meta(meta_val):
    if meta_val is None:
        return {}
    if isinstance(meta_val, bytes):
        try:
            meta_val = meta_val.decode("utf-8", errors="ignore")
        except Exception:
            return {}
    if isinstance(meta_val, str):
        try:
            return ast.literal_eval(meta_val)
        except Exception:
            # if it's plain JSON-like but not valid literal, return empty
            try:
                import json
                return json.loads(meta_val)
            except Exception:
                return {}
    if isinstance(meta_val, dict):
        return meta_val
    return {}


def strip_module_prefix(sd):
    new_sd = {}
    changed = False
    for k, v in sd.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
            changed = True
        else:
            new_sd[k] = v
    return new_sd, changed


def dump_checkpoint_info(ck, debug=False):
    info = {}
    if isinstance(ck, dict):
        info["keys"] = list(ck.keys())
        meta = ck.get("meta", None)
        info["meta"] = meta if meta is None or debug else ("<present>" if meta else "<empty>")
        sd = ck.get("state_dict", None)
        if sd is None and not any("weight" in k for k in ck.keys()):
            # probably plain state_dict saved directly
            sd = ck
        if isinstance(sd, dict):
            info["state_dict_keys"] = len(sd)
            # show first 8 keys/shapes
            small = {}
            count = 0
            for k, v in sd.items():
                if count >= 8:
                    break
                try:
                    small[k] = list(v.shape) if hasattr(v, "shape") else str(type(v))
                except Exception:
                    small[k] = str(type(v))
                count += 1
            info["state_dict_sample"] = small
    return info


def dump_h5_info(path, idx=0):
    info = {}
    with h5py.File(path, "r") as f:
        info["keys"] = list(f.keys())
        info["attrs"] = {k: (v if len(str(v)) < 200 else "<large>") for k, v in f.attrs.items()}
        ds_info = {}
        for k in f.keys():
            try:
                obj = f[k]
                if isinstance(obj, h5py.Dataset):
                    ds_info[k] = {"shape": tuple(obj.shape), "dtype": str(obj.dtype)}
                    # safe sample preview
                    try:
                        # avoid huge reads; read very small sample
                        sl = tuple(0 if d == 0 else min(1, d-1) for d in obj.shape)
                        # read the idx item if possible
                        if obj.ndim >= 1 and obj.shape[0] > idx:
                            sample = obj[idx:idx+1]
                        else:
                            sample = obj[0:1]
                        ds_info[k]["sample_shape"] = tuple(sample.shape)
                    except Exception:
                        ds_info[k]["sample_shape"] = "<preview_failed>"
                else:
                    ds_info[k] = {"type": str(type(obj))}
            except Exception as e:
                ds_info[k] = {"error": str(e)}
        info["datasets"] = ds_info
    return info


def infer_model_expected_in_channels(model):
    # Try to find first conv weight param to infer in_channels
    for name, param in model.named_parameters():
        if "conv" in name and param.dim() >= 2:
            # conv weight typical shapes:
            # Conv1d: (out_c, in_c, kernel)
            # Conv2d: (out_c, in_c, kH, kW)
            shape = tuple(param.shape)
            if len(shape) >= 2:
                return int(shape[1]), shape
    # fallback: look for any weight-looking param
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:
            return int(param.shape[1]), tuple(param.shape)
    return None, None


def ensure_input_shape(X, expected_in_channels=1, model_conv_ndim=1, debug=False):
    """
    Make X shape friendly for typical models:
      - want (N, C, L) for 1D conv models
      - or (N, C, H, W) for 2D conv models (here we try to place L into H or W)
    """
    X = np.array(X, dtype="float32")
    if debug:
        logging.info("Raw X shape: %s", X.shape)

    # collapse leading/trailing singleton dims
    while X.ndim > 1 and X.shape[0] == 1 and X.ndim > 3:
        # e.g., shape (1,1,L,1) -> squeeze leading singletons
        X = X.squeeze(axis=0)

    # If the dataset returned (N, L) -> add channel
    if X.ndim == 2:
        # shape (N, L) or (L,) when N==1
        if X.shape[0] == 1 and X.shape[1] != expected_in_channels:
            # it's (1,L) -> convert to (1,1,L)
            X = X.reshape((1, 1, X.shape[1]))
        else:
            X = X[:, None, :]
    elif X.ndim == 3:
        # possibilities: (N, C, L), (N, L, C)
        N, a, b = X.shape
        if a == expected_in_channels:
            pass  # likely correct (N,C,L)
        elif b == expected_in_channels:
            X = X.transpose((0, 2, 1))
        elif a == 1 and expected_in_channels != 1:
            # broadcast/warn
            logging.warning("Dataset has channel dim=1 but model expects %d channels", expected_in_channels)
        else:
            # ambiguous -> force add channel in middle
            X = X[:, None, :]
    elif X.ndim == 4:
        # maybe (N,1,H,W) or (N,H,W,1) etc.
        # try (N,C,H,W) -> if C==1 it's fine. If last dim==1, move it
        N, a, b, c = X.shape
        if a == expected_in_channels:
            pass
        elif c == expected_in_channels:
            X = X.transpose(0, 3, 1, 2)
        elif a == 1 and model_conv_ndim == 1:
            # reduce spatial dims into a single length axis
            rest = int(b * c)
            X = X.reshape((N, 1, rest))
        else:
            logging.warning("Ambiguous 4D input shape %s; attempting to flatten spatial dims for 1D conv", X.shape)
            rest = int(np.prod(X.shape[1:]))
            X = X.reshape((N, 1, rest))

    # final check
    if debug:
        logging.info("Post-processed X shape: %s", X.shape)
    return X


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pth checkpoint or state_dict")
    parser.add_argument("--h5", required=True, help="Path to HDF5 dataset")
    parser.add_argument("--idx", type=int, default=0, help="Index to infer")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    args = parser.parse_args()

    try:
        # 1) load checkpoint
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model file not found: {args.model}")
        ck = torch.load(args.model, map_location="cpu")
        ck_info = dump_checkpoint_info(ck, debug=args.debug)
        if args.debug:
            logging.info("Checkpoint info: %s", ck_info)

        # decide state_dict / meta
        if isinstance(ck, dict):
            state_dict = ck.get("state_dict", None)
            meta = ck.get("meta", {}) or {}
            if state_dict is None:
                # maybe they saved raw state_dict into a dict (no meta)
                # treat ck itself as sd if many keys look like params
                if any(isinstance(v, torch.Tensor) for v in ck.values()):
                    state_dict = ck
                    meta = {}
            if state_dict is None:
                raise RuntimeError("Unable to find state_dict in checkpoint.")
        else:
            # plain state_dict saved directly
            state_dict = ck
            meta = {}

        # if meta looks like bytes or str decode
        if isinstance(meta, (bytes, str)):
            meta = safe_eval_meta(meta)

        # attempt to strip module. prefix if needed
        orig_keys_sample = None
        if isinstance(state_dict, dict):
            orig_keys_sample = list(state_dict.keys())[:8]
            state_dict, changed = strip_module_prefix(state_dict)
            if changed and args.debug:
                logging.info("'module.' prefix removed from state_dict keys")

        # 2) import model class
        try:
            # try local import first
            from cnn_classifier import PaperCNN
        except Exception as e:
            # attempt to add parent dir to path
            script_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            if script_parent not in sys.path:
                sys.path.insert(0, script_parent)
            try:
                from cnn_classifier import PaperCNN
            except Exception:
                # final fallback: show helpful message
                logging.error("Failed to import cnn_classifier.PaperCNN. Make sure cnn_classifier.py is on PYTHONPATH.")
                raise

        n_faults = int(meta.get("n_faults", 16))
        mean = meta.get("mean", None)
        std = meta.get("std", None)
        if mean is not None:
            mean = np.array(mean, dtype="float32")
        if std is not None:
            std = np.array(std, dtype="float32")

        # instantiate model
        model = PaperCNN(in_channels=1, base_filters=32, num_classes=n_faults)
        # attempt load
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            # try stripping module prefixes again conservatively
            new_sd = {}
            for k, v in state_dict.items():
                new_sd[k.replace("module.", "")] = v
            model.load_state_dict(new_sd)

        model.eval()

        # infer expected in_channels
        expected_in_channels, conv_shape = infer_model_expected_in_channels(model)
        if args.debug:
            logging.info("Inferred model expected_in_channels=%s conv_shape=%s", expected_in_channels, conv_shape)

        # 3) open HDF5 and inspect
        if not os.path.exists(args.h5):
            raise FileNotFoundError(f"HDF5 file not found: {args.h5}")
        h5info = dump_h5_info(args.h5, idx=args.idx)
        if args.debug:
            logging.info("HDF5 info: %s", h5info)

        with h5py.File(args.h5, "r") as f:
            if "X" not in f:
                raise KeyError("HDF5 does not contain dataset 'X'.")
            if not isinstance(f["X"], h5py.Dataset):
                raise TypeError("f['X'] is not a dataset. Check file.")

            # Read sample; always produce a batch shape first dim N
            try:
                rawX = f["X"][args.idx: args.idx + 1]
            except Exception:
                # try to read first element gracefully
                rawX = f["X"][0:1]
            X = np.array(rawX, dtype="float32")

            # parse meta attr if present
            meta_attr = safe_eval_meta(f.attrs.get("meta", "{}"))
            fault_map = meta_attr.get("fault_label_map", {}) or {}
            # build reverse map robustly
            rev = {}
            for k, v in fault_map.items():
                try:
                    rev[int(v)] = str(k)
                except Exception:
                    rev[v] = k

            # severity and ur
            sev = None
            ur = None
            if "y_sev" in f:
                if isinstance(f["y_sev"], h5py.Dataset):
                    try:
                        sv = f["y_sev"][args.idx]
                        sev = int(np.array(sv).item())
                    except Exception:
                        sev = None
                else:
                    raise TypeError("f['y_sev'] exists but is not a dataset.")
            if "ur" in f:
                if isinstance(f["ur"], h5py.Dataset):
                    try:
                        urv = f["ur"][args.idx]
                        ur = float(np.array(urv).item())
                    except Exception:
                        ur = None
                else:
                    raise TypeError("f['ur'] exists but is not a dataset.")

        # shape inputs according to expected_in_channels
        if expected_in_channels is None:
            expected_in_channels = 1
        model_conv_ndim = 1 if (conv_shape is None or len(conv_shape) == 3) else 2
        X = ensure_input_shape(X, expected_in_channels=expected_in_channels, model_conv_ndim=model_conv_ndim, debug=args.debug)

        # normalization
        if mean is not None and std is not None:
            try:
                X = (X - mean) / (std + 1e-9)
            except Exception:
                # broadcasting attempts
                try:
                    # try (1, C, *) shape
                    mean_r = mean.reshape((1, mean.shape[0])) if mean.ndim == 1 else mean.reshape((1,) + mean.shape)
                    std_r = std.reshape((1, std.shape[0])) if std.ndim == 1 else std.reshape((1,) + std.shape)
                    X = (X - mean_r) / (std_r + 1e-9)
                except Exception:
                    logging.warning("Could not broadcast mean/std to X shape; skipping normalization")
        # convert to torch
        inp = torch.from_numpy(np.array(X)).float()
        # final dim sanity
        if inp.dim() == 2:
            inp = inp.unsqueeze(1)
        if args.debug:
            logging.info("Final tensor shape fed to model: %s", tuple(inp.shape))

        # inference
        with torch.no_grad():
            out_scores = model(inp)
            if out_scores.dim() == 1:
                out_scores = out_scores.unsqueeze(0)
            pred_idx = int(out_scores.argmax(dim=1).item())

        out = {"fault_id": pred_idx, "label": rev.get(pred_idx, str(pred_idx))}
        if sev is not None:
            out["severity"] = int(sev)
        if ur is not None:
            out["ur"] = float(ur)
        print("Prediction:", out)

    except Exception as exc:
        # print friendly and full traceback for debugging
        logging.error("Error during inference: %s", str(exc))
        tb = traceback.format_exc()
        print("\n--- FULL TRACEBACK (paste this if you need help) ---\n")
        print(tb)
        print("\n--- Helpful checks to paste next ---")
        print(f" * Run with --debug to print checkpoint/hdf5 summaries.")
        print(" * Confirm cnn_classifier.PaperCNN is importable: python -c \"from cnn_classifier import PaperCNN; print('ok')\"")
        print(f" * Confirm model file exists: {args.model}")
        print(f" * Confirm HDF5 file exists: {args.h5}")
        sys.exit(1)


if __name__ == "__main__":
    main()
