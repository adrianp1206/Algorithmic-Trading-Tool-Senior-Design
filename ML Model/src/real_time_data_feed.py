import time

# Assume `historical_data` is a list or array of stock price data points
def simulate_real_time_feed(env, model, historical_data, delay=1):
    """
    Simulate real-time prediction using historical data.

    Parameters:
        env: the custom RL environment for stock prediction
        model: the trained PPO model
        historical_data: the historical data to be fed as real-time data
        delay: time delay between steps to simulate real-time (in seconds)
    """
    state = env.reset()  # Reset environment to initial state
    done = False
    total_reward = 0
    step = 0

    print("Starting Real-Time Simulation...")

    # Loop through each data point as if it arrives in real-time
    for data_point in historical_data:
        if done:
            break
        
        # Simulate waiting for the next data point (optional delay)
        time.sleep(delay)
        
        # Get the model's action (weights) based on the current state
        action, _ = model.predict(state, deterministic=True)
        
        # Take a step in the environment
        next_state, reward, done, _ = env.step(action)
        
        # Print results for the current step
        print(f"Step {step}: Action={action}, Reward={reward}, Total Reward={total_reward}")
        
        # Update total reward and step counter
        total_reward += reward
        step += 1

        # Move to the next state
        state = next_state

    print("Real-Time Simulation Complete.")
    print(f"Total Reward from Real-Time Simulation: {total_reward}")
