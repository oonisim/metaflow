from metaflow import FlowSpec, step, batch, IncludeFile, Parameter, conda, conda_base


def get_python_version():
    """
    A convenience function to get the python version used to run this
    tutorial. This ensures that the conda environment is created with an
    available version of python.

    """
    import platform
    versions = {
        '2': '2.7.15',
        '3': '3.7.3'
    }
    return versions[platform.python_version_tuple()[0]]


@conda_base(python=get_python_version())
class TestGPUFlow(FlowSpec):

    @batch(cpu=2, gpu=1, memory=2400)
    @conda(libraries={'pytorch': '1.4.0'})
    @step
    def start(self):
        import os
        import sys
        import torch
        from subprocess import call
        print('__Python VERSION:', sys.version)
        print('__pyTorch VERSION:', torch.__version__)
        print('__CUDA VERSION')
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__Devices')
        call([
            "nvidia-smi",
            "--format=csv",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"
        ])
        print('Active CUDA Device: GPU', torch.cuda.current_device())
        print('Available devices ', torch.cuda.device_count())
        print('Current cuda device ', torch.cuda.current_device())

        print(f"GPU count: {torch.cuda.device_count()}")
        print(os.popen("nvidia-smi").read())

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    TestGPUFlow()
