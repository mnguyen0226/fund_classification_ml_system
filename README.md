# Fund Classification ML System
Mock ML system that is used classify whether a name is a Fund or Non-Fund. Note that we need to mock the data and we can use sklearn to train such model. We can train, evaluate, and pickle this model in a separate script.

I want to build a streamlit app that can use that model to classify a name to be Fund or Non-Fund.

We can save the mock data in CSV for easy to see. We don't have to use SQLite database.

Pain point:
- Design the system to retrain model with human feedback:
    - How to setup log for retrain.
    - How to design components.
    - How to setup our CSV so that we can periodically retrain model from scratch? 