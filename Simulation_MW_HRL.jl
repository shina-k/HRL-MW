using Plots, StatsBase, LinearAlgebra, DataFrames, CSV

#################################################################
# Implementation of a homeostatic reinforcement learning agent #
#################################################################

mutable struct HRLAgent
    α::Float64
    β::Float64
    m::Float64
    n::Float64
    w::Float64
    τ::Vector{Float64}
    homestatic_setpoint::Vector{Float64}
    homestatic_space::Vector{Float64}
    q::Vector{Float64}
    perseverance_q::Vector{Float64}
    post_action::Int64

    function HRLAgent(α::Float64, β::Float64,
                      m::Float64, n::Float64,w::Float64,
                       τ::Vector{Float64},
                      homestatic_setpoint::Vector{Float64},
                      homestatic_space::Vector{Float64},
                      post_action::Int64)
        n_ = length(homestatic_setpoint)
        q = Vector{Float64}(zeros(n_))
        perseverance_q = Vector{Float64}(zeros(n_))
        new(α, β, m, n, w, τ, homestatic_setpoint, homestatic_space, q,perseverance_q, post_action)
    end
end

#################################
# Implementation of environment #
#################################

mutable struct Environment

    observation_probability::Vector{Float64}
    observation_amount::Matrix{Float64}
    trial_time::Vector{Int64}
    time_step::Int64
    checker::Int64
    trans_prob::Array{Float64, 3}

    function Environment(observation_probability::Vector{Float64},
            observation_amount::Matrix{Float64},
            trial_time::Vector{Int64},
            time_step::Int64,checker::Int64,
            trans_prob::Array{Float64, 3})
        new(observation_probability, observation_amount,
        trial_time::Vector{Int64},time_step::Int64,checker::Int64,
        trans_prob::Array{Float64, 3})
    end
end

function step(env::Environment, action::Vector{Float64},stim::Int64)::Vector{Float64}
    n = length(action)
    
    reward_or_no = Vector{Float64}(rand(n) .< env.observation_probability)
    return reward_or_no .* env.observation_amount[stim,:] .* action # By multiplying one-hot action vector, set the reward amount of unchosen action to 0.
end

function present_stimuli(env::Environment)::Int64
    env.time_step += 1

    if env.time_step <= env.trial_time[1]
        stim = 1
    elseif env.time_step > env.trial_time[1]
        stim = 2
    end

    if env.time_step > sum(env.trial_time)
        stim = 1
        env.time_step = 1
        env.checker = 0
    end
    return stim
end

function softmax(agent::HRLAgent)::Vector{Float64}
    perseverance_adjust = agent.perseverance_q .- maximum(agent.perseverance_q)
    q_adjusted = agent.q .- maximum(agent.q) # To prevent overflow
    qp_adjusted = agent.w * q_adjusted + (1 - agent.w) * perseverance_adjust
    qp_exp = exp.(agent.β * qp_adjusted)
    return qp_exp / sum(qp_exp)
end

function choose_action(agent::HRLAgent, probs::Vector{Float64}, 
    stimulus::Int64, env::Environment)::Int64

    n = length(probs)

    if stimulus == 1
        probs = probs .* env.trans_prob[agent.post_action,:,1]

    elseif stimulus == 2
        if agent.post_action == 2 && env.checker == 0
            probs = probs .* env.trans_prob[agent.post_action,:,2]

        elseif agent.post_action == 2 && env.checker == 1
                probs = probs .* env.trans_prob[agent.post_action,:,1]
        else
            probs = probs .* env.trans_prob[agent.post_action,:,2]
        end
    end

    act = sample(1:(n+1), ProbabilityWeights(probs))
    agent.post_action = act

    if act == 1
        env.checker = 1
    end

    return act
end

function drive(agent::HRLAgent)::Float64
    return sum(abs.(agent.homestatic_setpoint .- agent.homestatic_space).^agent.n)^(1/agent.m)
end

function drive_next(agent::HRLAgent, observation::Vector{Float64})::Float64
    return sum((abs.(agent.homestatic_setpoint .- (1 .- agent.τ) .* agent.homestatic_space .- observation).^agent.n))^(1/agent.m)
end

function reward(agent::HRLAgent, observation::Vector{Float64})::Float64
    return drive(agent) - drive_next(agent, observation)
end

#insert function

function drive_each(agent::HRLAgent)::Vector{Float64}
    return abs.(agent.homestatic_setpoint .- agent.homestatic_space)
end

function drive_next_each(agent::HRLAgent, observation::Vector{Float64})::Vector{Float64}
    return (abs.(agent.homestatic_setpoint .- (1 .- agent.τ) .* agent.homestatic_space .- observation))
end

function reward_each(agent::HRLAgent, observation::Vector{Float64})::Vector{Float64}
    return drive_each(agent) - drive_next_each(agent, observation)
end

function calc_tau(agent::HRLAgent, observation::Vector{Float64})::Vector{Float64}
    return agent.homestatic_space .- (1 .- agent.τ) .* agent.homestatic_space
end
##

function update_homestatic_space(agent::HRLAgent, observation::Vector{Float64})
    agent.homestatic_space .= (1 .- agent.τ) .* agent.homestatic_space .+ observation
end

function updata_perseverance(agent::HRLAgent, action::Vector{Float64})
    δ = action .- agent.perseverance_q
    agent.perseverance_q .+= agent.α * δ
end 

function update_q(agent::HRLAgent, reward::Float64, action::Vector{Float64})
    δ = reward .- agent.q
    agent.q .+= agent.α * δ .* action # Multiplying δ by one-hot action vector, prevent update unchosen actions.
end

########################################
# Helper functions used in a simulation #
########################################

function as_onehot(k::Int64, n::Int64)::Vector{Float64}
    x = Matrix(I, n, n)
    return x[k, :]
end

# To allow flexibility in the variables to be returned, the return type was not set.
function step(agent::HRLAgent, env::Environment)
    stimulus = present_stimuli(env)
    p = softmax(agent)
    action = choose_action(agent, p, stimulus, env)
    obs = step(env, as_onehot(action, length(p)),stimulus)
    each = reward_each(agent, obs)
    r = reward(agent, obs)
    τ = calc_tau(agent,obs)
    δ = (r .- agent.q) .* as_onehot(action, length(p))
    updata_perseverance(agent, as_onehot(action, length(p)))
    update_q(agent, r, as_onehot(action, length(p)))
    update_homestatic_space(agent, obs)
    # For result output
    env.time_step, action, agent.q[1], agent.q[2], agent.q[3],agent.perseverance_q[1],agent.perseverance_q[2],agent.perseverance_q[3], agent.homestatic_space[1],agent.homestatic_space[2],agent.homestatic_space[3], stimulus, r, δ[1], δ[2], δ[3],τ[1],τ[2],τ[3],each[1],each[2],each[3]
end


###################################
# Run simulation and show results #
###################################

#############################
# export data for plot sim1 #
#############################
#with_mw
df = DataFrame()

for i in 1:100
  
env = Environment([1., 1., 1.], [1. 1. 1.; 1. 1. 1.], [30, 10], 0, 0,
    [[0. .5 .5]; [0. .5 .5]; [0. .5 .5];;;[0. .5 .5]; [.5 .25 .25]; [0. .5 .5]]) 
    
agent = HRLAgent(0.05, 10., 3., 4.,.95, [1/100, 1/100, 1/100], [30., 30., 0.], [0., 0., 0.], 2)

    trialN = 200
    result = map(_ -> step(agent, env), 1:(sum(env.trial_time)*trialN)) |> vcat |> DataFrame
    df = vcat(df,hcat(result,fill(i,size(result)[1])))
end

#no_mw
df = DataFrame()

for i in 1:100
  
env = Environment([1., 1., 1.], [1. 1. 1.; 1. 1. 1.], [30, 10], 0, 0,
    [[0. 1. 0.]; [0. 1. 0.]; [0. .5 .5];;;[0. 1. 0.]; [.5 .5 0.]; [0. .5 .5]])
    

agent = HRLAgent(0.05, 10., 3., 4.,.95, [1/100, 1/100, 1/100], [30., 30., 0.], [0., 0., 0.], 2)

    trialN = 200
    result = map(_ -> step(agent, env), 1:(sum(env.trial_time)*trialN)) |> vcat |> DataFrame
    df = vcat(df,hcat(result,fill(i,size(result)[1])))
end

###############################
# export data for plot sim2-1 #
###############################
#low_task
df = DataFrame()

for i in 1:100
  
env = Environment([1., 1., 1.], [1. 1. 1.; 1. 1. 1.], [30, 10], 0, 0,
    [[0. .5 .5]; [0. .5 .5]; [0. .5 .5];;;[0. .5 .5]; [.5 .25 .25]; [0. .5 .5]])
    
    
agent = HRLAgent(0.05, 10., 3., 4.,.95, [1/100, 1/100, 1/100], [5., 5., 0.], [0., 0., 0.], 2)

    trialN = 200
    result = map(_ -> step(agent, env), 1:(sum(env.trial_time)*trialN)) |> vcat |> DataFrame
    df = vcat(df,hcat(result,fill(i,size(result)[1])))
end

#mid_task
df = DataFrame()

for i in 1:100
  
env = Environment([1., 1., 1.], [1. 1. 1.; 1. 1. 1.], [30, 10], 0, 0,
    [[0. .5 .5]; [0. .5 .5]; [0. .5 .5];;;[0. .5 .5]; [.5 .25 .25]; [0. .5 .5]])
    
    
agent = HRLAgent(0.05, 10., 3., 4.,.95, [1/100, 1/100, 1/100], [30., 30., 0.], [0., 0., 0.], 2)

    trialN = 200
    result = map(_ -> step(agent, env), 1:(sum(env.trial_time)*trialN)) |> vcat |> DataFrame
    df = vcat(df,hcat(result,fill(i,size(result)[1])))
end

#high_task
df = DataFrame()

for i in 1:100
  
env = Environment([1., 1., 1.], [1. 1. 1.; 1. 1. 1.], [30, 10], 0, 0,
    [[0. .5 .5]; [0. .5 .5]; [0. .5 .5];;;[0. .5 .5]; [.5 .25 .25]; [0. .5 .5]])
    
    
agent = HRLAgent(0.05, 10., 3., 4.,.95, [1/100, 1/100, 1/100], [60., 60., 0.], [0., 0., 0.], 2)

    trialN = 200
    result = map(_ -> step(agent, env), 1:(sum(env.trial_time)*trialN)) |> vcat |> DataFrame
    df = vcat(df,hcat(result,fill(i,size(result)[1])))
end

###############################
# export data for plot sim2-2 #
###############################
#low_tau
df = DataFrame()

for i in 1:100
  
env = Environment([1., 1., 1.], [1. 1. 1.; 1. 1. 1.], [30, 10], 0, 0,
    [[0. .5 .5]; [0. .5 .5]; [0. .5 .5];;;[0. .5 .5]; [.5 .25 .25]; [0. .5 .5]])
    
    
agent = HRLAgent(0.05, 10., 3., 4.,.95, [1/130, 1/130, 1/100], [30., 30., 0.], [0., 0., 0.], 2)

    trialN = 200
    result = map(_ -> step(agent, env), 1:(sum(env.trial_time)*trialN)) |> vcat |> DataFrame
    df = vcat(df,hcat(result,fill(i,size(result)[1])))
end

#mid_tau
df = DataFrame()

for i in 1:100
  
env = Environment([1., 1., 1.], [1. 1. 1.; 1. 1. 1.], [30, 10], 0, 0,
    [[0. .5 .5]; [0. .5 .5]; [0. .5 .5];;;[0. .5 .5]; [.5 .25 .25]; [0. .5 .5]])
    
    
agent = HRLAgent(0.05, 10., 3., 4.,.95, [1/100, 1/100, 1/100], [30., 30., 0.], [0., 0., 0.], 2)

    trialN = 200
    result = map(_ -> step(agent, env), 1:(sum(env.trial_time)*trialN)) |> vcat |> DataFrame
    df = vcat(df,hcat(result,fill(i,size(result)[1])))
end

#high_tau
df = DataFrame()

for i in 1:100
  
env = Environment([1., 1., 1.], [1. 1. 1.; 1. 1. 1.], [30, 10], 0, 0,
    [[0. .5 .5]; [0. .5 .5]; [0. .5 .5];;;[0. .5 .5]; [.5 .25 .25]; [0. .5 .5]])
    
    
agent = HRLAgent(0.05, 10., 3., 4.,.95, [1/70, 1/70, 1/100], [30., 30., 0.], [0., 0., 0.], 2)

    trialN = 200
    result = map(_ -> step(agent, env), 1:(sum(env.trial_time)*trialN)) |> vcat |> DataFrame
    df = vcat(df,hcat(result,fill(i,size(result)[1])))
end

#############################
# export data for plot sim3 #
#############################
#low_mw
df = DataFrame()

for i in 1:100
  
env = Environment([1., 1., 1.], [1. 1. 1.; 1. 1. 1.], [30, 10], 0, 0,
    [[0. .5 .5]; [0. .5 .5]; [0. .5 .5];;;[0. .5 .5]; [.5 .25 .25]; [0. .5 .5]])
    
    
agent = HRLAgent(0.05, 10., 3., 4.,.95, [1/100, 1/100, 1/100], [30., 30., 0.], [0., 0., 0.], 2)

    trialN = 200
    result = map(_ -> step(agent, env), 1:(sum(env.trial_time)*trialN)) |> vcat |> DataFrame
    df = vcat(df,hcat(result,fill(i,size(result)[1])))
end

#mid_mw
df = DataFrame()

for i in 1:100
  
env = Environment([1., 1., 1.], [1. 1. 1.; 1. 1. 1.], [30, 10], 0, 0,
    [[0. .5 .5]; [0. .5 .5]; [0. .5 .5];;;[0. .5 .5]; [.5 .25 .25]; [0. .5 .5]])
    
    
agent = HRLAgent(0.05, 10., 3., 4.,.95, [1/100, 1/100, 1/100], [30., 30., 30.], [0., 0., 0.], 2)

    trialN = 200
    result = map(_ -> step(agent, env), 1:(sum(env.trial_time)*trialN)) |> vcat |> DataFrame
    df = vcat(df,hcat(result,fill(i,size(result)[1])))
end

#high_mw
df = DataFrame()

for i in 1:100
  
env = Environment([1., 1., 1.], [1. 1. 1.; 1. 1. 1.], [30, 10], 0, 0,
    [[0. .5 .5]; [0. .5 .5]; [0. .5 .5];;;[0. .5 .5]; [.5 .25 .25]; [0. .5 .5]])
    
    
agent = HRLAgent(0.05, 10., 3., 4.,.95, [1/100, 1/100, 1/100], [30., 30., 60.], [0., 0., 0.], 2)

    trialN = 200
    result = map(_ -> step(agent, env), 1:(sum(env.trial_time)*trialN)) |> vcat |> DataFrame
    df = vcat(df,hcat(result,fill(i,size(result)[1])))
end

###############################
# export data for plot sim4   #
###############################
#low_diffic
df = DataFrame()

for i in 1:100
  
env = Environment([1., 1., 1.], [1. 1. 1.; 1. 1. 1.], [30, 10], 0, 0,
    [[0. .5 .5]; [0. .5 .5]; [0. .5 .5];;;[0. .5 .5]; [.5 .25 .25]; [0. .5 .5]])
    
    
agent = HRLAgent(0.05, 10., 3., 4.,.95, [1/100, 1/100, 1/100], [30., 30., 0.], [0., 0., 0.], 2)

    trialN = 200
    result = map(_ -> step(agent, env), 1:(sum(env.trial_time)*trialN)) |> vcat |> DataFrame
    df = vcat(df,hcat(result,fill(i,size(result)[1])))
end

#mid_diffic
df = DataFrame()

for i in 1:100
  
env = Environment([1., 1., 1.], [1. .75 1.; 1. .75 1.], [30, 10], 0, 0,
    [[0. .5 .5]; [0. .5 .5]; [0. .5 .5];;;[0. .5 .5]; [.5 .25 .25]; [0. .5 .5]])
    
    
agent = HRLAgent(0.05, 10., 3., 4.,.95, [1/100, 1/100, 1/100], [30., 30., 0.], [0., 0., 0.], 2)

    trialN = 200
    result = map(_ -> step(agent, env), 1:(sum(env.trial_time)*trialN)) |> vcat |> DataFrame
    df = vcat(df,hcat(result,fill(i,size(result)[1])))
end

#high_diffic
df = DataFrame()

for i in 1:100
  
env = Environment([1., 1., 1.], [1. .1 1.; 1. .1 1.], [30, 10], 0, 0,
    [[0. .5 .5]; [0. .5 .5]; [0. .5 .5];;;[0. .5 .5]; [.5 .25 .25]; [0. .5 .5]])
    
    
agent = HRLAgent(0.05, 10., 3., 4.,.95, [1/100, 1/100, 1/100], [30., 30., 0.], [0., 0., 0.], 2)

    trialN = 200
    result = map(_ -> step(agent, env), 1:(sum(env.trial_time)*trialN)) |> vcat |> DataFrame
    df = vcat(df,hcat(result,fill(i,size(result)[1])))
end

#too_diffic
df = DataFrame()

for i in 1:100
  
env = Environment([1., .1, 1.], [1. 0. 1.; 1. 0. 1.], [30, 10], 0, 0,
    [[0. .5 .5]; [0. .5 .5]; [0. .5 .5];;;[0. .5 .5]; [.5 .25 .25]; [0. .5 .5]])
    
    
agent = HRLAgent(0.05, 10., 3., 4.,.95, [1/100, 1/100, 1/100], [30., 30., 0.], [0., 0., 0.], 2)

    trialN = 200
    result = map(_ -> step(agent, env), 1:(sum(env.trial_time)*trialN)) |> vcat |> DataFrame
    df = vcat(df,hcat(result,fill(i,size(result)[1])))
end

