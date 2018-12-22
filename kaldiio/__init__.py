# flake8: noqa F401
import pkg_resources

from kaldiio.matio import load_ark
from kaldiio.matio import load_mat
from kaldiio.matio import load_scp
from kaldiio.matio import save_ark
from kaldiio.matio import save_mat
from kaldiio.highlevel import parse_specifier
from kaldiio.highlevel import ReadHelper
from kaldiio.highlevel import WriteHelper
from kaldiio.utils import open_like_kaldi
from kaldiio.wavio import load_wav_scp

try:
    __version__ = pkg_resources.get_distribution('kaldiio').version
except Exception:
    __version__ = None
del pkg_resources
