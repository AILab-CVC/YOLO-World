# from https://github.com/open-mmlab/mmengine/blob/main/mmengine/dist/utils.py
# from https://github.com/open-mmlab/mmengine/blob/main/mmengine/dist/dist.py
# Apache License 2.0
import datetime
import functools
import numpy as np
import os
import os.path as osp
import pickle
import shutil
import subprocess
import tempfile
import torch
import torch.multiprocessing as mp
from collections.abc import Iterable, Mapping
from itertools import chain, zip_longest
from torch import Tensor
from torch import distributed as torch_dist
from torch.distributed import ProcessGroup
from typing import Any, Callable, List, Optional, Tuple, Union

_LOCAL_PROCESS_GROUP = None


def is_distributed() -> bool:
    """Return True if distributed environment has been initialized."""
    return torch_dist.is_available() and torch_dist.is_initialized()


def get_local_group() -> Optional[ProcessGroup]:
    """Return local process group."""
    if not is_distributed():
        return None

    if _LOCAL_PROCESS_GROUP is None:
        raise RuntimeError('Local process group is not created, please use '
                           '`init_local_group` to setup local process group.')

    return _LOCAL_PROCESS_GROUP


def get_default_group() -> Optional[ProcessGroup]:
    """Return default process group."""

    return torch_dist.distributed_c10d._get_default_group()


def infer_launcher():
    if 'WORLD_SIZE' in os.environ:
        return 'pytorch'
    elif 'SLURM_NTASKS' in os.environ:
        return 'slurm'
    elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        return 'mpi'
    else:
        return 'none'


def init_dist(launcher, backend='nccl', init_backend='torch', **kwargs) -> None:
    """Initialize distributed environment.

    Args:
        launcher (str): Way to launcher multi processes. Supported launchers
            are 'pytorch', 'mpi' and 'slurm'.
        backend (str): Communication Backends. Supported backends are 'nccl',
            'gloo' and 'mpi'. Defaults to 'nccl'.
        **kwargs: keyword arguments are passed to ``init_process_group``.
    """
    timeout = kwargs.get('timeout', None)
    if timeout is not None:
        # If a timeout (in seconds) is specified, it must be converted
        # to a timedelta object before forwarding the call to
        # the respective backend, because they expect a timedelta object.
        try:
            kwargs['timeout'] = datetime.timedelta(seconds=timeout)
        except TypeError as exception:
            raise TypeError(f'Timeout for distributed training must be provided as '
                            f"timeout in seconds, but we've received the type "
                            f'{type(timeout)}. Please specify the timeout like this: '
                            f"dist_cfg=dict(backend='nccl', timeout=1800)") from exception
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, init_backend=init_backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, init_backend=init_backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, init_backend='torch', **kwargs) -> None:
    """Initialize distributed environment with PyTorch launcher.

    Args:
        backend (str): Backend of torch.distributed. Supported backends are
            'nccl', 'gloo' and 'mpi'. Defaults to 'nccl'.
        **kwargs: keyword arguments are passed to ``init_process_group``.
    """
    rank = int(os.environ['RANK'])
    # LOCAL_RANK is set by `torch.distributed.launch` since PyTorch 1.1
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(local_rank)

    if init_backend == 'torch':
        torch_dist.init_process_group(backend=backend, **kwargs)
    elif init_backend == 'deepspeed':
        import deepspeed  # TODO
        deepspeed.init_distributed(dist_backend=backend, **kwargs)
    elif init_backend == 'colossalai':
        import colossalai  # TODO
        colossalai.launch_from_torch(backend=backend, **kwargs)
    else:
        raise ValueError('supported "init_backend" is "torch" or "deepspeed", '
                         f'but got {init_backend}')


def _init_dist_mpi(backend, **kwargs) -> None:
    """Initialize distributed environment with MPI launcher.

    Args:
        backend (str): Backend of torch.distributed. Supported backends are
            'nccl', 'gloo' and 'mpi'. Defaults to 'nccl'.
        **kwargs: keyword arguments are passed to ``init_process_group``.
    """
    if backend == 'smddp':
        try:
            import smdistributed.dataparallel.torch.torch_smddp  # noqa: F401
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError('Please use an Amazon SageMaker DLC to access smdistributed: '
                                      'https://github.com/aws/deep-learning-containers/blob/master'
                                      '/available_images.md#sagemaker-framework-containers'
                                      '-sm-support-only') from e
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    if 'MASTER_PORT' not in os.environ:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    if 'MASTER_ADDR' not in os.environ:
        raise KeyError('The environment variable MASTER_ADDR is not set')
    os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    torch_dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend, port=None, init_backend='torch', **kwargs) -> None:
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    # Not sure when this environment variable could be None, so use a fallback
    local_rank_env = os.environ.get('SLURM_LOCALID', None)
    if local_rank_env is not None:
        local_rank = int(local_rank_env)
    else:
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
    addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(proc_id)

    torch.cuda.set_device(local_rank)

    if init_backend == 'torch':
        torch_dist.init_process_group(backend=backend, **kwargs)
    elif init_backend == 'deepspeed':
        import deepspeed
        deepspeed.init_distributed(dist_backend=backend, **kwargs)
    elif init_backend == 'colossalai':
        import colossalai
        colossalai.launch_from_slurm(
            backend=backend,
            host=os.environ['MASTER_ADDR'],
            port=os.environ['MASTER_PORT'],
            **kwargs,
        )
    else:
        raise ValueError('supported "init_backend" is "torch" or "deepspeed", '
                         f'but got {init_backend}')


def init_local_group(node_rank: int, num_gpus_per_node: int):
    """Setup the local process group.

    Setup a process group which only includes processes that on the same
    machine as the current process.

    The code is modified from
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py

    Args:
        node_rank (int): Rank of machines used for training.
        num_gpus_per_node (int): Number of gpus used for training in a single
            machine.
    """  # noqa: W501
    global _LOCAL_PROCESS_GROUP
    assert _LOCAL_PROCESS_GROUP is None

    ranks = list(range(node_rank * num_gpus_per_node, (node_rank + 1) * num_gpus_per_node))
    _LOCAL_PROCESS_GROUP = torch_dist.new_group(ranks)


def get_backend(group: Optional[ProcessGroup] = None) -> Optional[str]:
    """Return the backend of the given process group.

    Note:
        Calling ``get_backend`` in non-distributed environment will return
        None.

    Args:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific
            group is specified, the calling process must be part of
            :attr:`group`. Defaults to None.

    Returns:
        str or None: Return the backend of the given process group as a lower
        case string if in distributed environment, otherwise None.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_backend(group)
    else:
        return None


def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    """Return the number of the given process group.

    Note:
        Calling ``get_world_size`` in non-distributed environment will return
        1.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the number of processes of the given process group if in
        distributed environment, otherwise 1.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_world_size(group)
    else:
        return 1


def get_rank(group: Optional[ProcessGroup] = None) -> int:
    """Return the rank of the given process group.

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Note:
        Calling ``get_rank`` in non-distributed environment will return 0.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the rank of the process group if in distributed
        environment, otherwise 0.
    """

    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_rank(group)
    else:
        return 0


def get_local_size() -> int:
    """Return the number of the current node.

    Returns:
        int: Return the number of processes in the current node if in
        distributed environment, otherwise 1.
    """
    if not is_distributed():
        return 1

    if _LOCAL_PROCESS_GROUP is None:
        raise RuntimeError('Local process group is not created, please use '
                           '`init_local_group` to setup local process group.')

    return torch_dist.get_world_size(_LOCAL_PROCESS_GROUP)


def get_local_rank() -> int:
    """Return the rank of current process in the current node.

    Returns:
        int: Return the rank of current process in the current node if in
        distributed environment, otherwise 0
    """
    if not is_distributed():
        return 0

    if _LOCAL_PROCESS_GROUP is None:
        raise RuntimeError('Local process group is not created, please use '
                           '`init_local_group` to setup local process group.')

    return torch_dist.get_rank(_LOCAL_PROCESS_GROUP)


def get_dist_info(group: Optional[ProcessGroup] = None) -> Tuple[int, int]:
    """Get distributed information of the given process group.

    Note:
        Calling ``get_dist_info`` in non-distributed environment will return
        (0, 1).

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        tuple[int, int]: Return a tuple containing the ``rank`` and
        ``world_size``.
    """
    world_size = get_world_size(group)
    rank = get_rank(group)
    return rank, world_size


def is_main_process(group: Optional[ProcessGroup] = None) -> bool:
    """Whether the current rank of the given process group is equal to 0.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        bool: Return True if the current rank of the given process group is
        equal to 0, otherwise False.
    """
    return get_rank(group) == 0


def master_only(func: Callable) -> Callable:
    """Decorate those methods which should be executed in master process.

    Args:
        func (callable): Function to be decorated.

    Returns:
        callable: Return decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)

    return wrapper


def barrier(group: Optional[ProcessGroup] = None) -> None:
    """Synchronize all processes from the given process group.

    This collective blocks processes until the whole group enters this
    function.

    Note:
        Calling ``barrier`` in non-distributed environment will do nothing.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        torch_dist.barrier(group)


def get_data_device(data: Union[Tensor, Mapping, Iterable]) -> torch.device:
    """Return the device of ``data``.

    If ``data`` is a sequence of Tensor, all items in ``data`` should have a
    same device type.

    If ``data`` is a dict whose values are Tensor, all values should have a
    same device type.

    Args:
        data (Tensor or Sequence or dict): Inputs to be inferred the device.

    Returns:
        torch.device: The device of ``data``.

    Examples:
        >>> import torch
        >>> from mmengine.dist import cast_data_device
        >>> # data is a Tensor
        >>> data = torch.tensor([0, 1])
        >>> get_data_device(data)
        device(type='cpu')
        >>> # data is a list of Tensor
        >>> data = [torch.tensor([0, 1]), torch.tensor([2, 3])]
        >>> get_data_device(data)
        device(type='cpu')
        >>> # data is a dict
        >>> data = {'key1': torch.tensor([0, 1]), 'key2': torch.tensor([0, 1])}
        >>> get_data_device(data)
        device(type='cpu')
    """
    if isinstance(data, Tensor):
        return data.device
    elif isinstance(data, Mapping):
        pre = None
        for v in data.values():
            cur = get_data_device(v)
            if pre is None:
                pre = cur
            else:
                if cur != pre:
                    raise ValueError('device type in data should be consistent, but got '
                                     f'{cur} and {pre}')
        if pre is None:
            raise ValueError('data should not be empty.')
        return pre
    elif isinstance(data, Iterable) and not isinstance(data, str):
        pre = None
        for item in data:
            cur = get_data_device(item)
            if pre is None:
                pre = cur
            else:
                if cur != pre:
                    raise ValueError('device type in data should be consistent, but got '
                                     f'{cur} and {pre}')
        if pre is None:
            raise ValueError('data should not be empty.')
        return pre
    else:
        raise TypeError('data should be a Tensor, sequence of tensor or dict, '
                        f'but got {data}')


def get_comm_device(group: Optional[ProcessGroup] = None) -> torch.device:
    """Return the device for communication among groups.

    Args:
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        torch.device: The device of backend.
    """
    backend = get_backend(group)
    if backend == 'hccl':
        import torch_npu  # noqa: F401
        return torch.device('npu', torch.npu.current_device())
    elif backend == torch_dist.Backend.NCCL:
        return torch.device('cuda', torch.cuda.current_device())
    elif backend == 'cncl':
        import torch_mlu  # noqa: F401
        return torch.device('mlu', torch.mlu.current_device())
    elif backend == 'smddp':
        return torch.device('cuda', torch.cuda.current_device())
    elif backend == 'mccl':
        import torch_musa
        return torch.device('musa', torch_musa.current_device())
    else:
        # GLOO and MPI backends use cpu device by default
        return torch.device('cpu')


def cast_data_device(data: Union[Tensor, Mapping, Iterable],
                     device: torch.device,
                     out: Optional[Union[Tensor, Mapping, Iterable]] = None) -> Union[Tensor, Mapping, Iterable]:
    """Recursively convert Tensor in ``data`` to ``device``.

    If ``data`` has already on the ``device``, it will not be casted again.

    Args:
        data (Tensor or list or dict): Inputs to be casted.
        device (torch.device): Destination device type.
        out (Tensor or list or dict, optional): If ``out`` is specified, its
            value will be equal to ``data``. Defaults to None.

    Returns:
        Tensor or list or dict: ``data`` was casted to ``device``.
    """
    if out is not None:
        if type(data) != type(out):
            raise TypeError('out should be the same type with data, but got data is '
                            f'{type(data)} and out is {type(data)}')

        if isinstance(out, set):
            raise TypeError('out should not be a set')

    if isinstance(data, Tensor):
        if get_data_device(data) == device:
            data_on_device = data
        else:
            data_on_device = data.to(device)

        if out is not None:
            # modify the value of out inplace
            out.copy_(data_on_device)  # type: ignore

        return data_on_device
    elif isinstance(data, Mapping):
        data_on_device = {}
        if out is not None:
            data_len = len(data)
            out_len = len(out)  # type: ignore
            if data_len != out_len:
                raise ValueError('length of data and out should be same, '
                                 f'but got {data_len} and {out_len}')

            for k, v in data.items():
                data_on_device[k] = cast_data_device(v, device, out[k])  # type: ignore
        else:
            for k, v in data.items():
                data_on_device[k] = cast_data_device(v, device)

        if len(data_on_device) == 0:
            raise ValueError('data should not be empty')

        # To ensure the type of output as same as input, we use `type(data)`
        # to wrap the output
        return type(data)(data_on_device)  # type: ignore
    elif isinstance(data, Iterable) and not isinstance(data, str) and not isinstance(data, np.ndarray):
        data_on_device = []
        if out is not None:
            for v1, v2 in zip(data, out):
                data_on_device.append(cast_data_device(v1, device, v2))
        else:
            for v in data:
                data_on_device.append(cast_data_device(v, device))

        if len(data_on_device) == 0:
            raise ValueError('data should not be empty')

        return type(data)(data_on_device)  # type: ignore
    else:
        raise TypeError('data should be a Tensor, list of tensor or dict, '
                        f'but got {data}')


def sync_random_seed(group: Optional[ProcessGroup] = None) -> int:
    """Synchronize a random seed to all processes.
    """
    seed = np.random.randint(2**31)
    if get_world_size(group) == 1:
        return seed

    if group is None:
        group = get_default_group()

    backend_device = get_comm_device(group)

    if get_rank(group) == 0:
        random_num = torch.tensor(seed, dtype=torch.int32).to(backend_device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32).to(backend_device)

    torch_dist.broadcast(random_num, src=0, group=group)

    return random_num.item()


def broadcast(data: Tensor, src: int = 0, group: Optional[ProcessGroup] = None) -> None:
    """Broadcast the data from ``src`` process to the whole group.

    ``data`` must have the same number of elements in all processes
    participating in the collective.

    Note:
        Calling ``broadcast`` in non-distributed environment does nothing.

    Args:
        data (Tensor): Data to be sent if ``src`` is the rank of current
            process, and data to be used to save received data otherwise.
        src (int): Source rank. Defaults to 0.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Examples:
        >>> import torch
        >>> import mmengine.dist as dist

        >>> # non-distributed environment
        >>> data = torch.arange(2, dtype=torch.int64)
        >>> data
        tensor([0, 1])
        >>> dist.broadcast(data)
        >>> data
        tensor([0, 1])

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> data = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> data
        tensor([1, 2]) # Rank 0
        tensor([3, 4]) # Rank 1
        >>> dist.broadcast(data)
        >>> data
        tensor([1, 2]) # Rank 0
        tensor([1, 2]) # Rank 1
    """
    if get_world_size(group) > 1:
        if group is None:
            group = get_default_group()

        input_device = get_data_device(data)
        backend_device = get_comm_device(group)
        data_on_device = cast_data_device(data, backend_device)
        # broadcast requires tensor is contiguous
        data_on_device = data_on_device.contiguous()  # type: ignore
        torch_dist.broadcast(data_on_device, src, group)

        if get_rank(group) != src:
            cast_data_device(data_on_device, input_device, data)


def _object_to_tensor(obj: Any) -> Tuple[Tensor, Tensor]:
    """Serialize picklable python object to tensor."""
    byte_storage = torch.ByteStorage.from_buffer(pickle.dumps(obj))
    # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor
    # and specifying dtype. Otherwise, it will cause 100X slowdown.
    # See: https://github.com/pytorch/pytorch/issues/65696
    byte_tensor = torch.ByteTensor(byte_storage)
    local_size = torch.LongTensor([byte_tensor.numel()])
    return byte_tensor, local_size


def _tensor_to_object(tensor: Tensor, tensor_size: int) -> Any:
    """Deserialize tensor to picklable python object."""
    buf = tensor.cpu().numpy().tobytes()[:tensor_size]
    return pickle.loads(buf)


def all_gather_object(data: Any, group: Optional[ProcessGroup] = None) -> List[Any]:
    """Gather picklable objects from the whole group into a list. Similar to
    :func:`all_gather`, but Python objects can be passed in. Note that the
    object must be picklable in order to be gathered.

    Note:
        Calling ``all_gather_object`` in non-distributed environment does
        nothing and just returns a list containing :attr:`data` itself.

    Note:
        Unlike PyTorch ``torch.distributed.all_gather_object``,
        :meth:`all_gather_object` in MMEngine does not pass in an empty list
        ``gather_list`` and returns the ``gather_list`` directly, which is
        more convenient. The difference between their interfaces is as below:

        - MMEngine: all_gather_object(data, group) -> gather_list
        - PyTorch: all_gather_object(gather_list, data, group) -> None

    Args:
        data (Any): Pickable Python object to be broadcast from current
            process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        list[Tensor]: Return a list containing data from the whole group if
        in distributed environment, otherwise a list only containing
        :attr:`data` itself.

    Note:
        For NCCL-based process groups, internal tensor representations
        of objects must be moved to the GPU device before communication starts.
        In this case, the used device is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is correctly set so that each rank has an individual
        GPU, via ``torch.cuda.set_device()``.

    Examples:
        >>> import torch
        >>> import mmengine.dist as dist

        >>> # non-distributed environment
        >>> data = ['foo', 12, {1: 2}]  # any picklable object
        >>> gather_objects = dist.all_gather_object(data[dist.get_rank()])
        >>> output
        ['foo']

        >>> # distributed environment
        >>> # We have 3 process groups, 3 ranks.
        >>> output = dist.all_gather_object(data[dist.get_rank()])
        >>> output
        ['foo', 12, {1: 2}]  # Rank 0
        ['foo', 12, {1: 2}]  # Rank 1
        ['foo', 12, {1: 2}]  # Rank 2
    """
    world_size = get_world_size(group)
    if world_size == 1:
        return [data]

    if group is None:
        group = get_default_group()

    gather_list = [None] * world_size

    torch_dist.all_gather_object(gather_list, data, group)

    return gather_list


def collect_results(results: list, size: int, device: str = 'cpu', tmpdir: Optional[str] = None) -> Optional[list]:
    """Collected results in distributed environments.

    Args:
        results (list[object]): Result list containing result parts to be
            collected. Each item of ``result_part`` should be a picklable
            object.
        size (int): Size of the results, commonly equal to length of
            the results.
        device (str): Device name. Optional values are 'cpu', 'gpu' or 'npu'.
        tmpdir (str | None): Temporal directory for collected results to
            store. If set to None, it will create a temporal directory for it.
            ``tmpdir`` should be None when device is 'gpu' or 'npu'.
            Defaults to None.

    Returns:
        list or None: The collected results.

    Examples:
        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> import mmengine.dist as dist
        >>> if dist.get_rank() == 0:
                data = ['foo', {1: 2}]
            else:
                data = [24, {'a': 'b'}]
        >>> size = 4
        >>> output = dist.collect_results(data, size, device='cpu')
        >>> output
        ['foo', 24, {1: 2}, {'a': 'b'}]  # rank 0
        None  # rank 1
    """
    if device not in ['gpu', 'cpu', 'npu']:
        raise NotImplementedError(f"device must be 'cpu' , 'gpu' or 'npu', but got {device}")

    if device == 'gpu' or device == 'npu':
        assert tmpdir is None, f'tmpdir should be None when device is {device}'
        return _collect_results_device(results, size)
    else:
        return collect_results_cpu(results, size, tmpdir)


def collect_results_cpu(result_part: list, size: int, tmpdir: Optional[str] = None) -> Optional[list]:
    """Collect results under cpu mode.

    On cpu mode, this function will save the results on different gpus to
    ``tmpdir`` and collect them by the rank 0 worker.

    Args:
        result_part (list): Result list containing result parts
            to be collected. Each item of ``result_part`` should be a picklable
            object.
        size (int): Size of the results, commonly equal to length of
            the results.
        tmpdir (str | None): Temporal directory for collected results to
            store. If set to None, it will create a random temporal directory
            for it. Defaults to None.

    Returns:
        list or None: The collected results.

    Examples:
        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> import mmengine.dist as dist
        >>> if dist.get_rank() == 0:
                data = ['foo', {1: 2}]
            else:
                data = [24, {'a': 'b'}]
        >>> size = 4
        >>> output = dist.collect_results_cpu(data, size)
        >>> output
        ['foo', 24, {1: 2}, {'a': 'b'}]  # rank 0
        None  # rank 1
    """
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]

    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ), 32, dtype=torch.uint8)
        if rank == 0:
            # mmengine.mkdir_or_exist('.dist_test')
            os.makedirs('.dist_test', exist_ok=True)
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(bytearray(tmpdir.encode()), dtype=torch.uint8)
            dir_tensor[:len(tmpdir)] = tmpdir
        broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.numpy().tobytes().decode().rstrip()
    else:
        os.makedirs(tmpdir, exist_ok=True)
        # mmengine.mkdir_or_exist(tmpdir)

    # dump the part result to the dir
    with open(osp.join(tmpdir, f'part_{rank}.pkl'), 'wb') as f:  # type: ignore
        pickle.dump(result_part, f, protocol=2)

    barrier()

    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            path = osp.join(tmpdir, f'part_{i}.pkl')  # type: ignore
            if not osp.exists(path):
                raise FileNotFoundError(f'{tmpdir} is not an shared directory for '
                                        f'rank {i}, please make sure {tmpdir} is a shared '
                                        'directory for all ranks!')
            with open(path, 'rb') as f:
                part_list.append(pickle.load(f))
        # sort the results
        ordered_results = []
        zipped_results = zip_longest(*part_list)
        ordered_results = [i for i in chain.from_iterable(zipped_results) if i is not None]
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)  # type: ignore
        return ordered_results


def _collect_results_device(result_part: list, size: int) -> Optional[list]:
    """Collect results under gpu or npu mode."""
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]

    # gather all result part. Note that NCCL does not support gather so use
    # all_gather_object instead.
    part_list = all_gather_object(result_part)

    if rank == 0:
        # sort the results
        ordered_results = []
        zipped_results = zip_longest(*part_list)
        ordered_results = [i for i in chain.from_iterable(zipped_results) if i is not None]
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
    else:
        return None


def collect_results_gpu(result_part: list, size: int) -> Optional[list]:
    """Collect results under gpu mode.

    On gpu mode, this function will encode results to gpu tensors and use gpu
    communication for results collection.

    Args:
        result_part (list[object]): Result list containing result parts
            to be collected. Each item of ``result_part`` should be a picklable
            object.
        size (int): Size of the results, commonly equal to length of
            the results.

    Returns:
        list or None: The collected results.

    Examples:
        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> import mmengine.dist as dist
        >>> if dist.get_rank() == 0:
                data = ['foo', {1: 2}]
            else:
                data = [24, {'a': 'b'}]
        >>> size = 4
        >>> output = dist.collect_results_gpu(data, size)
        >>> output
        ['foo', 24, {1: 2}, {'a': 'b'}]  # rank 0
        None  # rank 1
    """
    return _collect_results_device(result_part, size)


def _to_cpu(data: Any) -> Any:
    """transfer all tensors and BaseDataElement to cpu."""
    if isinstance(data, Tensor):
        return data.to('cpu')
    elif isinstance(data, list):
        return [_to_cpu(d) for d in data]
    elif isinstance(data, tuple):
        return tuple(_to_cpu(d) for d in data)
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    else:
        return data


def broadcast_object_list(data: List[Any], src: int = 0, group: Optional[ProcessGroup] = None) -> None:
    """Broadcasts picklable objects in ``object_list`` to the whole group.
    Similar to :func:`broadcast`, but Python objects can be passed in. Note
    that all objects in ``object_list`` must be picklable in order to be
    broadcasted.

    Note:
        Calling ``broadcast_object_list`` in non-distributed environment does
        nothing.

    Args:
        data (List[Any]): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank
            will be broadcast, but each rank must provide lists of equal sizes.
        src (int): Source rank from which to broadcast ``object_list``.
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        device (``torch.device``, optional): If not None, the objects are
            serialized and converted to tensors which are moved to the
            ``device`` before broadcasting. Default is ``None``.

    Note:
        For NCCL-based process groups, internal tensor representations of
        objects must be moved to the GPU device before communication starts.
        In this case, the used device is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is correctly set so that each rank has an individual
        GPU, via ``torch.cuda.set_device()``.

    Examples:
        >>> import torch
        >>> import mmengine.dist as dist

        >>> # non-distributed environment
        >>> data = ['foo', 12, {1: 2}]
        >>> dist.broadcast_object_list(data)
        >>> data
        ['foo', 12, {1: 2}]

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 3.
        >>>     data = ["foo", 12, {1: 2}]  # any picklable object
        >>> else:
        >>>     data = [None, None, None]
        >>> dist.broadcast_object_list(data)
        >>> data
        ["foo", 12, {1: 2}]  # Rank 0
        ["foo", 12, {1: 2}]  # Rank 1
    """
    assert isinstance(data, list)

    if get_world_size(group) > 1:
        if group is None:
            group = get_default_group()

        torch_dist.broadcast_object_list(data, src, group)
