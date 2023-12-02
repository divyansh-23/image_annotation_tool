# Import necessary functions and classes from other modules
from ui import main_ui
from database import setup_database
from session_state import initialize_session_state

if __name__ == "__main__":
    setup_database()
    initialize_session_state()
    main_ui()
