import pytest

from banditry import AgentConfig, OFUGPConfig, TSConfig, build_agent
from banditry.agents import ModelEnum, OFUGPAgent, TSAgent
from banditry.sampling_oracles import FeelGoodNLL, LangevinSampler


def test_build_ofugp_gp(numeric_space):
    agent = build_agent(OFUGPConfig(surrogate="gp", noise_std_proxy=1.0), numeric_space)
    assert isinstance(agent, OFUGPAgent)
    assert agent.surrogate is ModelEnum.gp


def test_build_ofugp_svgp(numeric_space):
    agent = build_agent(OFUGPConfig(surrogate="svgp", noise_std_proxy=1.0), numeric_space)
    assert isinstance(agent, OFUGPAgent)
    assert agent.surrogate is ModelEnum.svgp


def test_build_ts_langevin(numeric_space):
    agent = build_agent(TSConfig(sampler="langevin"), numeric_space)
    assert isinstance(agent, TSAgent)
    assert agent.sampler_cls is LangevinSampler
    assert agent.nll is None


def test_build_ts_feel_good(numeric_space):
    agent = build_agent(TSConfig(sampler="langevin", feel_good=True), numeric_space)
    assert isinstance(agent.nll, FeelGoodNLL)


def test_build_ts_nuts(numeric_space):
    pytest.importorskip("pyro")
    from banditry.sampling_oracles import NUTSSampler

    agent = build_agent(TSConfig(sampler="nuts"), numeric_space)
    assert agent.sampler_cls is NUTSSampler


def test_unknown_sampler_raises(numeric_space):
    with pytest.raises(ValueError, match="Unknown sampler"):
        build_agent(TSConfig(sampler="bogus"), numeric_space)


def test_unknown_config_raises(numeric_space):
    with pytest.raises(TypeError, match="Unknown config"):
        build_agent(AgentConfig(), numeric_space)
