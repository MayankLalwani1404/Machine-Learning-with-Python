# Advanced Rock Paper Scissors player that adapts to different opponent strategies
import random
from collections import Counter

def player(prev_play, opponent_history=[], my_history=[], game_count=[0]):
    # Reset on new match
    if prev_play == "" and len(opponent_history) > 0:
        opponent_history.clear()
        my_history.clear()
        game_count[0] = 0
    
    opponent_history.append(prev_play)
    game_count[0] += 1
    
    # Define what beats what
    beats = {'R': 'P', 'P': 'S', 'S': 'R'}
    guess = "R"  # Default
    
    # Try each strategy and see which one to use
    if game_count[0] <= 5:
        # Start with mixed strategy to gather info
        guess = ["R", "P", "S", "R", "P"][game_count[0] - 1]
    else:
        # Determine opponent type based on patterns
        opp_type = determine_opponent_type(opponent_history, my_history, game_count[0])
        
        if opp_type == "quincy":
            guess = counter_quincy_v2(game_count[0])
        elif opp_type == "kris":
            guess = counter_kris_v2(my_history)
        elif opp_type == "abbey":
            guess = counter_abbey_v2(opponent_history, my_history)
        else:  # mrugesh or unknown
            guess = counter_mrugesh_v2(my_history)
    
    my_history.append(guess)
    return guess

def determine_opponent_type(opponent_history, my_history, game_count):
    """Determine opponent type based on behavior patterns"""
    clean_opp = [move for move in opponent_history if move != ""]
    
    if len(clean_opp) < 5:
        return "abbey"  # Default to Abbey early on
    
    # Test for Quincy (fixed pattern R, R, P, P, S) - very specific pattern
    if test_quincy_pattern(clean_opp):
        return "quincy"
    
    # Test for Kris (counters our previous move) - very specific behavior
    if test_kris_pattern(opponent_history, my_history):
        return "kris"
    
    # Test for Mrugesh (frequency analysis) - only if very confident
    if game_count > 20 and test_mrugesh_behavior_strong(opponent_history, my_history):
        return "mrugesh"
    
    # Default to Abbey for everything else
    return "abbey"

def test_quincy_pattern(clean_history):
    """Test if opponent follows Quincy's R,R,P,P,S pattern"""
    if len(clean_history) < 6:
        return False
    
    pattern = ["R", "R", "P", "P", "S"]
    recent = clean_history[-10:]  # Look at last 10 moves
    
    best_match = 0
    for start_offset in range(len(pattern)):
        matches = 0
        for i, move in enumerate(recent):
            expected = pattern[(i + start_offset) % len(pattern)]
            if move == expected:
                matches += 1
        best_match = max(best_match, matches)
    
    return best_match >= len(recent) * 0.8  # 80% match

def test_kris_pattern(opponent_history, my_history):
    """Test if opponent counters our previous moves"""
    if len(opponent_history) < 4 or len(my_history) < 3:
        return False
    
    beats = {'R': 'P', 'P': 'S', 'S': 'R'}
    matches = 0
    total = 0
    
    for i in range(1, min(8, len(opponent_history))):
        if (opponent_history[-i] and opponent_history[-i] != "" and 
            len(my_history) > i):
            expected = beats.get(my_history[-(i+1)], 'R')
            if opponent_history[-i] == expected:
                matches += 1
            total += 1
    
    return total > 0 and matches / total >= 0.7

def test_mrugesh_behavior(opponent_history, my_history):
    """Test if opponent behaves like Mrugesh (counters our most frequent move)"""
    if len(opponent_history) < 8 or len(my_history) < 7:
        return False
    
    beats = {'R': 'P', 'P': 'S', 'S': 'R'}
    
    # Look at our moves and see if opponent counters our most frequent
    matches = 0
    total = 0
    
    for i in range(7, min(15, len(my_history))):
        # Get our moves up to position i
        our_moves_so_far = my_history[:i]
        if len(our_moves_so_far) >= 3:
            # Find our most frequent move up to that point
            from collections import Counter
            counter = Counter(our_moves_so_far)
            most_frequent = counter.most_common(1)[0][0]
            
            # Check if opponent played the counter to our most frequent
            expected_counter = beats[most_frequent]
            if i < len(opponent_history) and opponent_history[i] == expected_counter:
                matches += 1
            total += 1
    
    return total > 0 and matches / total >= 0.6

def test_mrugesh_behavior_strong(opponent_history, my_history):
    """Stronger test for Mrugesh behavior - requires higher confidence"""
    if len(opponent_history) < 12 or len(my_history) < 11:
        return False
    
    beats = {'R': 'P', 'P': 'S', 'S': 'R'}
    matches = 0
    total = 0
    
    # Look at more recent moves for stronger evidence
    for i in range(10, min(20, len(my_history))):
        our_moves_so_far = my_history[:i]
        if len(our_moves_so_far) >= 5:
            from collections import Counter
            counter = Counter(our_moves_so_far)
            most_frequent = counter.most_common(1)[0][0]
            
            expected_counter = beats[most_frequent]
            if i < len(opponent_history) and opponent_history[i] == expected_counter:
                matches += 1
            total += 1
    
    return total > 0 and matches / total >= 0.8  # Higher threshold

def counter_quincy_v2(game_count):
    """Counter Quincy's fixed R,R,P,P,S pattern"""
    pattern = ["R", "R", "P", "P", "S"]
    beats = {'R': 'P', 'P': 'S', 'S': 'R'}
    quincy_next = pattern[(game_count - 1) % len(pattern)]
    return beats[quincy_next]

def counter_kris_v2(my_history):
    """Counter Kris who counters our last move"""
    if not my_history:
        return "R"
    beats = {'R': 'P', 'P': 'S', 'S': 'R'}
    # Kris will counter our last move, so we counter his counter
    kris_next = beats[my_history[-1]]
    return beats[kris_next]

def counter_mrugesh_v2(my_history):
    """Counter Mrugesh who counters our most frequent move"""
    if len(my_history) < 3:
        return random.choice(["R", "P", "S"])
    
    beats = {'R': 'P', 'P': 'S', 'S': 'R'}
    
    # Look at our last 10 moves
    recent_moves = my_history[-10:] if len(my_history) >= 10 else my_history
    counter = Counter(recent_moves)
    most_frequent = counter.most_common(1)[0][0]
    
    # Mrugesh will counter our most frequent, so we counter that
    mrugesh_next = beats[most_frequent]
    return beats[mrugesh_next]

def counter_abbey_v2(opponent_history, my_history):
    """Counter Abbey by predicting what Abbey will predict and then countering that"""
    beats = {'R': 'P', 'P': 'S', 'S': 'R'}
    
    if len(my_history) < 2:
        return "R"
    
    # Exact simulation of Abbey's algorithm
    play_order = {
        "RR": 0, "RP": 0, "RS": 0,
        "PR": 0, "PP": 0, "PS": 0,
        "SR": 0, "SP": 0, "SS": 0,
    }
    
    # Count all our 2-move patterns (exactly like Abbey does)
    for i in range(len(my_history) - 1):
        last_two = my_history[i] + my_history[i + 1]
        if last_two in play_order:
            play_order[last_two] += 1
    
    # Get our current last move (what Abbey sees)
    prev_opponent_play = my_history[-1] if my_history else "R"
    
    # Abbey's prediction logic
    potential_plays = [
        prev_opponent_play + "R",
        prev_opponent_play + "P",
        prev_opponent_play + "S",
    ]
    
    sub_order = {
        k: play_order[k]
        for k in potential_plays if k in play_order
    }
    
    if not sub_order or max(sub_order.values()) == 0:
        # No patterns, Abbey defaults to countering Rock
        abbey_prediction = "R"
    else:
        # Abbey predicts based on most frequent pattern
        prediction_pattern = max(sub_order, key=sub_order.get)
        abbey_prediction = prediction_pattern[-1]  # Last character
    
    # Abbey plays counter to its prediction
    abbey_move = beats[abbey_prediction]
    
    # We play counter to Abbey's move
    return beats[abbey_move]
