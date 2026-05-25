"""
Monkey-patch mmcv.ops with pure-PyTorch implementations.

Import this BEFORE any mmrotate import to avoid mmcv._ext DLL load failure.
"""
import sys
import importlib
from types import ModuleType


def _make_fake_ext():
    """Create a fake mmcv._ext module with stubs for ALL CUDA functions
    that mmcv.ops modules try to import. This prevents ImportError
    when mmcv.ops.__init__ loads its submodules."""
    from importlib.machinery import ModuleSpec
    from importlib.abc import Loader

    class _FakeLoader(Loader):
        def create_module(self, spec): return ModuleType(spec.name)
        def exec_module(self, module): pass

    ext = ModuleType('mmcv._ext')
    ext.__file__ = __file__
    ext.__spec__ = ModuleSpec('mmcv._ext', _FakeLoader())
    ext.__loader__ = _FakeLoader()

    class _FakeFn:
        """Fake CUDA function that does nothing (actual impl is in our pure-PyTorch ops)."""
        def __init__(self, name=''):
            self.__name__ = name
        def __call__(self, *args, **kwargs):
            return None
        def __repr__(self):
            return f'<FakeCUDAFn: {self.__name__}>'

    # Monkey-patch __getattr__ to return a fake fn for any requested name
    ext.__class__ = type('_FakeExtModule', (ModuleType,), {
        '__getattr__': lambda s, name: _FakeFn(name)
    })

    # Also set some well-known attributes that might be accessed directly
    for name in ('active_rotated_filter_forward', 'active_rotated_filter_backward',
                 'ActiveRotatedFilterForward', 'ActiveRotatedFilterBackward',
                 'box_iou_rotated_forward', 'box_iou_rotated_backward',
                 'nms_rotated_forward', 'nms_rotated_backward',
                 'deform_conv_forward', 'deform_conv_backward',
                 'deform_conv_ext', 'roi_align_forward', 'roi_align_backward',
                 'carafe_forward', 'carafe_backward', 'CARAFEnaive',
                 'corner_pool_forward', 'corner_pool_backward',
                 'psamask_forward', 'psamask_backward',
                 'rotated_feature_align_forward', 'rotated_feature_align_backward'):
        setattr(ext, name, _FakeFn(name))

    sys.modules['mmcv._ext'] = ext
    return ext


def patch():
    """Inject pure-PyTorch ops into mmcv.ops.

    Must be called BEFORE mmrotate (or anything that does 'from mmcv.ops import ...')
    to avoid the CUDA extension load failure.
    """
    if 'mmcv._ext' in sys.modules and hasattr(sys.modules['mmcv._ext'], '__patched__'):
        return  # Already patched

    _make_fake_ext()

    # Now mmcv.ops import will succeed (fake _ext handles all requests)
    import mmcv.ops
    import mmcv.ops.nms

    # Replace CUDA ops with our pure-PyTorch implementations
    from atrumod.ops.rotated_iou import box_iou_rotated
    from atrumod.ops.rotated_nms import nms_rotated, batched_nms
    from atrumod.ops.active_rotated_filter import active_rotated_filter
    from atrumod.ops.deform_conv import DeformConv2d

    mmcv.ops.box_iou_rotated = box_iou_rotated
    mmcv.ops.nms_rotated = nms_rotated
    mmcv.ops.batched_nms = batched_nms
    mmcv.ops.active_rotated_filter = active_rotated_filter
    mmcv.ops.DeformConv2d = DeformConv2d
    mmcv.ops.nms.batched_nms = batched_nms

    sys.modules['mmcv._ext'].__patched__ = True
    return True


patch()
