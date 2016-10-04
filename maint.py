import os
import warnings

def rebuild_test_data(force=False):
    warnings.warn('Make sure you really want to do this. Pass the force=True option'
                  " if you're sure", stacklevel=2)
    if not force:
        return
    else:
        warnings.warn('Proceeding to overwrite tests...')
    from hlm.both_levels.tests import make_data as bltests
    from hlm.upper_level.tests import make_data as ultests
    from hlm.lower_level.tests import make_data as lltests
    from hlm.tests import make_data as overall
    for module in [bltests, ultests, lltests, overall]:
        module.build()

if __name__ == '__main__':
    warnings.warn('cowardly refusing to wipe out tests. If you want to update test data,'
                  'start up an interpreter, import the maint module, '
                  'and run `rebuild_test_data`', stacklevel=2)
