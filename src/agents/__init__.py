import importlib

def get_runner(hps):
    Runner = importlib.import_module(f'src.agents.{hps.agent.name}').Runner
    return Runner(hps)
