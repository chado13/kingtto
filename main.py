import numpy as np

def main():
    one_hots = np.load("app/data/lotto_one_hot.npy")
    print(one_hots.shape)


if __name__ == "__main__":
    main()
