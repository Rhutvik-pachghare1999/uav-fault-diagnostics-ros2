import os
import shutil
from glob import glob
from pathlib import Path

from setuptools import setup

package_name = 'uav_aegis'
VENDOR_FILES = ("ros2_inference_node.py", "cnn_classifier.py", "px4_log_replay.py")


def vendor_scripts():
    """Copy inference scripts into the package so the install tree is
    self-contained (no parents[3] repo-relative sys.path hacks).

    The vendored copies keep their own `sys.path.insert(0, <own dir>)`, so
    `from cnn_classifier import PaperCNN` resolves to the vendored module.
    """
    repo = Path(os.environ.get("UAV_AEGIS_REPO_ROOT",
                                Path(__file__).resolve().parents[3]))
    scripts = repo / "scripts"
    if not scripts.is_dir():
        raise SystemExit(
            f"cannot find repo scripts dir {scripts}; run colcon build from "
            f"the repository, or set UAV_AEGIS_REPO_ROOT")
    dest = Path(__file__).resolve().parent / package_name / "vendor"
    dest.mkdir(parents=True, exist_ok=True)
    init = dest / "__init__.py"
    if not init.exists():
        init.write_text("")
    for name in VENDOR_FILES:
        src = scripts / name
        if not src.is_file():
            raise SystemExit(f"missing script to vendor: {src}")
        shutil.copy2(src, dest / name)


vendor_scripts()

setup(
    name=package_name,
    version='0.2.0',
    packages=[package_name, package_name + '.vendor'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Rhutvik Pachghare',
    maintainer_email='rhutvik@example.com',
    description='UAV Fault Diagnostics with Deep Learning',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fault_inference_node = uav_aegis.fault_inference_node:main',
            'px4_log_replay = uav_aegis.px4_log_replay:main',
        ],
    },
)
