from fabric.decorators import task
import six
if six.PY3:
    from . import module_fabtasks as tasks
else:
    import module_fabtasks as tasks

@task
def foo():
    pass

def bar():
    pass
