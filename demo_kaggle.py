#!/usr/bin/env python3
"""
Demo script for processing Kaggle tickets with configurable limits.
Use this to test the system with different numbers of tickets.
"""

import sys
from main import main
from config import DEFAULT_TICKET_LIMIT

def run_kaggle_demo(ticket_limit: int = None):
    """
    Run the Kaggle ticket processing demo with a specific ticket limit.
    
    Args:
        ticket_limit: Number of tickets to process. If None, uses DEFAULT_TICKET_LIMIT.
    """
    if ticket_limit is not None:
        # Temporarily override the default limit
        import config
        original_limit = config.DEFAULT_TICKET_LIMIT
        config.DEFAULT_TICKET_LIMIT = ticket_limit
        
        try:
            print(f"🎯 Running demo with {ticket_limit} tickets")
            main()
        finally:
            # Restore original limit
            config.DEFAULT_TICKET_LIMIT = original_limit
    else:
        print(f"🎯 Running demo with default limit ({DEFAULT_TICKET_LIMIT} tickets)")
        main()

if __name__ == "__main__":
    # Check if user provided a ticket limit
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
            if limit <= 0:
                print("❌ Ticket limit must be a positive number")
                sys.exit(1)
            run_kaggle_demo(limit)
        except ValueError:
            print("❌ Invalid ticket limit. Please provide a number.")
            sys.exit(1)
    else:
        # Show usage and run with default
        print("Usage: python demo_kaggle.py [number_of_tickets]")
        print(f"Example: python demo_kaggle.py 10  # Process 10 tickets")
        print(f"         python demo_kaggle.py     # Process {DEFAULT_TICKET_LIMIT} tickets (default)")
        print()
        run_kaggle_demo()