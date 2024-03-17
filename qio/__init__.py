from qio import qio

def QIO(mf, sol, act_space=None, logger=None, log_file='qio.log'):
    return qio.QIO(mf, sol, act_space, logger, log_file)