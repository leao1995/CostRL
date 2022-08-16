import importlib

def get_runner(hps):
    Runner = importlib.import_module(f'.{hps.agent.name}').Runner
    return Runner(hps)
