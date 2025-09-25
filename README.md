# Machine-Learning-with-Python

This repository contains all the projects required to get the Machine Learning with Python certification.

Project 1: Rock, Paper, Scissors AI bot

Description:
This project implements an advanced Rock Paper Scissors AI that can defeat four different opponent strategies with a win rate of at least 60% against each.

Solution Strategy

Opponent Detection System
The bot uses pattern analysis to identify which of four opponent types it's facing:

1. Quincy: Follows a fixed repeating pattern (R, R, P, P, S)
2. Kris: Always plays the counter to the player's previous move
3. Mrugesh: Analyzes the player's move frequency and counters the most common move
4. Abbey: Uses sophisticated Markov chain analysis of 2-move patterns to predict the next move

Counter Strategies

•  vs Quincy: Predicts the next move in the fixed sequence and plays the counter
•  vs Kris: Plays what beats Kris's counter to our previous move (counter-counter strategy)
•  vs Mrugesh: Tracks our own move frequency and counters Mrugesh's predicted counter
•  vs Abbey: Simulates Abbey's pattern analysis algorithm to predict what Abbey thinks we'll play, then counters Abbey's counter move

Key Implementation Features

•  Adaptive Opponent Recognition: Uses behavioral pattern matching to classify opponents
•  State Management: Properly handles game state between matches using mutable default parameters
•  Fallback Strategies: Robust handling when opponent classification is uncertain
•  Pattern Analysis: Implements frequency analysis and Markov chain simulation

Results
The AI successfully achieves 60%+ win rates against all four opponent types, with some strategies achieving 80-99% win rates.
