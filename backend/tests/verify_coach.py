
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from coaches.realtime_coach import RealtimeCoach

def test_coach():
    print("--- Testing RealtimeCoach ---")
    coach = RealtimeCoach()
    
    # 1. Test Text Stability (Caching)
    print("\n1. Testing Text Stability:")
    error_type = "elbow_stability"
    
    # First call: Should generate a tip
    tip1 = coach.get_text_feedback("bicep_curls", error_type)
    print(f"Frame 1 Tip: '{tip1}'")
    
    # Second call (same error): Should be IDENTICAL (Cached)
    tip2 = coach.get_text_feedback("bicep_curls", error_type)
    print(f"Frame 2 Tip: '{tip2}'")
    
    if tip1 == tip2:
        print("✅ SUCCESS: Tip remained stable (Cached).")
    else:
        print("❌ FAILURE: Tip changed! flickering detected.")
        
    # Third call (different error): Should change
    tip3 = coach.get_text_feedback("bicep_curls", "curl_higher")
    print(f"Frame 3 New Error Tip: '{tip3}'")
    
    if tip1 != tip3:
        print("✅ SUCCESS: Tip updated for new error.")
    else:
        print("❌ FAILURE: Tip failed to update.")

    # 2. Test Post-Rep Command (Merged Logic)
    print("\n2. Testing Post-Rep Command (Merged Logic):")
    # Simulate a bad rep
    form_states = ["ELBOW_DRIFT", "BODY_SWING"]
    cmd = coach.get_post_rep_command("bicep_curls", form_states)
    print(f"Post-Rep Command: '{cmd}'")
    
    if cmd:
        print("✅ SUCCESS: Generated post-rep command from merged logic.")
    else:
        print("❌ FAILURE: Failed to generate post-rep command.")

    # 3. Test Arrow Feedback
    print("\n3. Testing Arrow Feedback:")
    arrows = coach.get_arrow_feedback("bicep_curls", tip3)
    print(f"Arrows for '{tip3}': {arrows}")
    
    if len(arrows) > 0 and arrows[0]['type'] == 'curl_higher':
        print("✅ SUCCESS: Arrow feedback generated.")
    else:
        print("❌ FAILURE: Arrow feedback missing or incorrect.")

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    test_coach()
