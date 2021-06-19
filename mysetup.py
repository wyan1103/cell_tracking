from distutils.core import setup
import py2exe
import matplotlib

opts = {
    'py2exe': {
        'includes': ['matplotlib.backends.backend_tkagg']
    }
}
setup(options=opts, data_files=matplotlib.get_py2exe_datafiles(), console=['cell_detector.py'])