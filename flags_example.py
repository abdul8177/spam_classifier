from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("model_name", None, "Name of the Model to be used")
flags.DEFINE_integer("num_train_epochs", 10, "Number of epochs to be used")
flags.DEFINE_string("optimizer", None, "Name of the optimizer to be used")

def main(args):
    del args
    for _ in range(0, FLAGS.num_train_epochs):
        print(FLAGS.model_name, FLAGS.optimizer)

if __name__=="__main__":
    app.run(main)
