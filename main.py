from conllu_reader import ConlluReader
from algorithm import ArcEager
from algorithm import Sample
from model import ParserMLP  
from postprocessor import PostProcessor
from conll18_ud_eval import load_conllu_file, evaluate

def read_file(reader, path, inference):
    trees = reader.read_conllu_file(path, inference)
    print(f"Read a total of {len(trees)} sentences from {path}")
    print (f"Printing the first sentence of the training set... trees[0] = {trees[0]}")
    for token in trees[0]:
        print (token)
    print ()
    return trees


"""
ALREADY IMPLEMENTED
Read and convert CoNLLU files into tree structures
"""
# Initialize the ConlluReader
reader = ConlluReader()
train_trees = read_file(reader,path="en_partut-ud-train_clean.conllu", inference=False)
dev_trees = read_file(reader,path="en_partut-ud-dev_clean.conllu", inference=False)
test_trees = read_file(reader,path="en_partut-ud-test_clean.conllu", inference=True)


"""
We remove the non-projective sentences from the training and development set,
as the Arc-Eager algorithm cannot parse non-projective sentences.

We don't remove them from test set set, because for those we only will do inference
"""
train_trees = reader.remove_non_projective_trees(train_trees)
dev_trees = reader.remove_non_projective_trees(dev_trees)

print ("Total training trees after removing non-projective sentences", len(train_trees))
print ("Total dev trees after removing non-projective sentences", len(dev_trees))

#Create and instance of the ArcEager
arc_eager = ArcEager()

print ("\n ------ TODO: Implement the rest of the assignment ------")

# TODO: Complete the ArcEager algorithm class.
# 1. Implement the 'oracle' function and auxiliary functions to determine the correct parser actions.
#    Note: The SHIFT action is already implemented as an example.
#    Additional Note: The 'create_initial_state()', 'final_state()', and 'gold_arcs()' functions are already implemented.
# 2. Use the 'oracle' function in ArcEager to generate all training samples, creating a dataset for training the neural model.

print('-- Generando training samples --')
train_samples = []
for train_tree in train_trees:
    tree = arc_eager.oracle(train_tree)
    train_samples.extend(tree)

# 3. Utilize the same 'oracle' function to generate development samples for model tuning and evaluation.
print('-- Generando dev samples --')
dev_samples = []
for dev_tree in dev_trees:
    tree = arc_eager.oracle(dev_tree)
    dev_samples.extend(tree)

# TODO: Implement the 'state_to_feats' function in the Sample class.
# This function should convert the current parser state into a list of features for use by the neural model classifier.
## No lo ejecutamos aquí ya que se utiliza posteriormente en la función de entreno del modelo.

# TODO: Define and implement the neural model in the 'model.py' module.
# 1. Train the model on the generated training datSaset.
print('-- Entrenando el modelo --')
parser = ParserMLP(epochs = 45, parada_temprana = True)
parser.train(train_samples, dev_samples)

# 2. Evaluate the model's performance using the development dataset.
dev_accuracy = parser.evaluate(dev_samples)

# 3. Conduct inference on the test set with the trained model.
test_results = parser.run(test_trees)

# 4. Save the parsing results of the test set in CoNLLU format for further analysis.
# Guardamos los resultados de inferencia con test en un archivo Conllu

with open("parsed_output.conllu", "w", encoding="utf-8") as file:
    for sentence in test_results:
        tokens = sentence[1:] if sentence[0].form.upper() == "ROOT" else sentence
        
        for i, token in enumerate(tokens):
            token.id = i + 1

        for token in tokens:
            if token.head != 0:
                try:
                    old_head = int(token.head)
                except Exception:
                    old_head = 0
                token.head = old_head if 0 < old_head <= len(tokens) else 0

        roots = [t for t in tokens if t.head == 0]
        if len(roots) > 1:
            main_root = roots[0]
            for t in roots[1:]:
                t.head = main_root.id

        for token in tokens:
            token_str = str(token)
            # Reemplazamos directamente el carácter problemático
            token_str = token_str.replace("â€“", "–")
            token_str = token_str.replace("Ã´", "ô")
            token_str = token_str.replace("Ã©", "é")
            file.write(token_str + "\n")
        file.write("\n")

# Evaluamos este archivo usando conll18_ud_eval
gold_ud = load_conllu_file("en_partut-ud-test_clean.conllu")
pred_ud = load_conllu_file("parsed_output.conllu")
metrics = evaluate(gold_ud, pred_ud)
print("LAS F1 Score: {:.2f}%".format(100 * metrics["LAS"].f1))
print("UAS F1 Score: {:.2f}%".format(100 * metrics["UAS"].f1))


# # TODO: Utilize the 'postprocessor' module (already implemented).
print('\n-- Utilizando postprocessor --')
# 1. Read the output saved in the CoNLLU file and address any issues with ill-formed trees.
p = PostProcessor()
corrupted_trees = reader.read_conllu_file("parsed_output.conllu") 

# 2. Specify the file path: path = "<YOUR_PATH_TO_OUTPUT_FILE>"
path = "uncorrupted_output.conllu"

# 3. Process the file: trees = postprocessor.postprocess(path)S
uncorrupted_trees = p.postprocess("parsed_output.conllu")

# 4. Save the processed trees to a new output file.
with open(path, "w", encoding="utf-8") as file:
    for tree in uncorrupted_trees:
        for token in tree:
            file.write(str(token) + "\n")
        file.write("\n")