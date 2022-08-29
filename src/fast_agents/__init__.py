import importlib

def get_runner(hps):
    Runner = importlib.import_module(f'src.fast_agents.{hps.agent.name}').Runner
    return Runner(hps)