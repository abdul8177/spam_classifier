from absl import flags
from absl import app

FLAGS = flags.FLAGS


flags.DEFINE_integer(
    "train_batch_size", None, "Total batch size for training"
)

flags.DEFINE_integer(
    "train_epochs", None, "Total batch size for training"
)

flags.DEFINE_float(
    "train_dropout", None, "Dropout layer "
)

flags.DEFINE_string(
    "train_optimizer", None,
    "Optimizer to be used for model training.")

flags.DEFINE_string(
    "train_loss_fn", None,
    "Loss function to be used for model training.")

flags.DEFINE_string(
    "dense_layer_activation", None,
    "Loss function to be used for model training.")


flags.DEFINE_string(
    "output_layer_activation", None,
    "Loss function to be used for model training.")
    

def main(argv):
    print(FLAGS.train_batch_size)
    print(FLAGS.output_layer_activation)
    print(FLAGS.dense_layer_activation)
    print(FLAGS.train_optimizer)
    print(FLAGS.train_dropout)
    print(FLAGS.train_epochs)

if __name__ == "__main__":
    app.run(main)