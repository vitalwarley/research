from .fiw import FIW


class FIWFaCoRNet(FIW):

    TRAIN_PAIRS = "facornet/train_sort_A2_m.txt"
    VAL_PAIRS_MODEL_SEL = "facornet/val_choose_A.txt"
    VAL_PAIRS_THRES_SEL = "facornet/val_A.txt"
    TEST_PAIRS = "facornet/test_A.txt"

    # AdaFace uses BGR -- should I revert conversion read_image here?

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


if __name__ == "__main__":
    fiw = FIW(root_dir="../../datasets/", sample_path=FIWFaCoRNet.TRAIN_PAIRS)
