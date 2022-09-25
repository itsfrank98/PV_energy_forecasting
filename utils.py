import pickle

def save_to_pickle(name, c):
    with open(name, 'wb') as f:
        pickle.dump(c, f)

def load_from_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
