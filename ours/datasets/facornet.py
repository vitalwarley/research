from .fiw import FIW


class FIWFaCoRNet(FIW):

    TRAIN_PAIRS = "txt/train_sort_A2_m.txt"
    VAL_PAIRS_MODEL_SEL = "txt/val_choose_A.txt"
    VAL_PAIRS_THRES_SEL = "txt/val_A.txt"
    TEST_PAIRS = "txt/test_A.txt"

    # AdaFace uses BGR -- should I revert conversion read_image here?

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


if __name__ == "__main__":
    fiw = FIW(root_dir="../../datasets/", sample_path=FIWFaCoRNet.TRAIN_PAIRS)
