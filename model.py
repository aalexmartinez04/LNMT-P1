from conllu_token import Token
import tensorflow as tf
import numpy as np
from algorithm import ArcEager, Sample, Transition
import matplotlib.pyplot as plt
import pandas as pd

arc_eager = ArcEager()
plt.switch_backend('Agg')

class ParserMLP:
    """
    A Multi-Layer Perceptron (MLP) class for a dependency parser, using TensorFlow and Keras.

    This class implements a neural network model designed to predict transitions in a dependency 
    parser. It utilizes the Keras Functional API, which is more suited for multi-task learning scenarios 
    like this one. The network is trained to map parsing states to transition actions, facilitating 
    the parsing process in natural language processing tasks.

    Attributes:
        word_emb_dim (int): Dimensionality of the word embeddings. Defaults to 100.
        hidden_dim (int): Dimension of the hidden layer in the neural network. Defaults to 64.
        epochs (int): Number of training epochs. Defaults to 1.
        batch_size (int): Size of the batches used in training. Defaults to 64.

    Methods:
        train(training_samples, dev_samples): Trains the MLP model using the provided training and 
            development samples. It maps these samples to IDs that can be processed by an embedding 
            layer and then calls the Keras compile and fit functions.

        evaluate(samples): Evaluates the performance of the model on a given set of samples. The 
            method aims to assess the accuracy in predicting both the transition and dependency types, 
            with expected accuracies ranging between 75% and 85%.

        run(sents): Processes a list of sentences (tokens) using the trained model to perform dependency 
            parsing. This method implements the vertical processing of sentences to predict parser 
            transitions for each token.

        Feel free to add other parameters and functions you might need to create your model
    """

    def __init__(self, word_emb_dim: int = 100, hidden_dim: int = 128, 
                 epochs: int = 10, batch_size: int = 64, parada_temprana: bool = True):
        """
        Initializes the ParserMLP class with the specified dimensions and training parameters.

        Parameters:
            word_emb_dim (int): The dimensionality of the word embeddings.
            hidden_dim (int): The size of the hidden layer in the MLP.
            epochs (int): The number of epochs for training the model.
            batch_size (int): The batch size used during model training.
        """
        self._word_emb_dim = word_emb_dim
        self._hidden_dim = hidden_dim
        self._epochs = epochs
        self._batch_size = batch_size
        self._parada_temprana = parada_temprana

        self._transitions = {"LEFT-ARC":0,
                             "RIGHT-ARC":1,
                             "REDUCE":2,
                             "SHIFT":3}

        self._feature_vocab = {}
        self._dependency_vocab = {"NULL": 0} 

        self.train_history = []
        self.dev_history = []

    def create_vocab(self, tree_samples):

        # Convertimos cada estado en una lista de características
        sentence_features = [sample.state_to_feats(2, 2) for sample in tree_samples]

        # Obtenemos el vocabulario de palabras y etiquetas 
        for state in sentence_features:
            for word in state:
                if word not in self._feature_vocab:
                    self._feature_vocab[word] = len(self._feature_vocab)

        # Obtenemos el vocabulario de dependencias
        for sample in tree_samples:
            dep = sample.transition.dependency if sample.transition.action in [ArcEager.LA, ArcEager.RA] and sample.transition.dependency is not None else "NULL"
            if dep not in self._dependency_vocab:
                self._dependency_vocab[dep] = len(self._dependency_vocab)

        return self._feature_vocab, self._dependency_vocab
              

    def get_samples(self, tree_samples):
        
        dependencies = []
        transitions = []

        # Sacamos la matriz de características utilizando self._feature_vocab para obtener los índices asociados
        features_matrix = self.get_indices(tree_samples)
        
        for sample in tree_samples:
            # Sacamos la lista de transiciones 
            transitions.append(self._transitions[sample.transition.action])
            
            dep = sample.transition.dependency if sample.transition.action in [ArcEager.LA, ArcEager.RA] and sample.transition.dependency is not None else "NULL"
            
            # Sacamos la lista de dependencias utilizando self._dependency_vocab para obtener los índices asociados
            dependencies.append(self._dependency_vocab[dep])   
        
        y_action = np.array(transitions)
        y_dep = np.array(dependencies)
        
        # print(features_matrix, y_action, y_dep)
        # print(features_matrix.shape, len(y_action), len(y_dep))
        return features_matrix, y_action, y_dep
    
    def get_indices(self, tree_samples):
        features = []
        sentence_features = [sample.state_to_feats(2, 2) for sample in tree_samples]

        for state in sentence_features:
            features_state = [self._feature_vocab.get(word, len(self._feature_vocab)+1) for word in state]
            features.append(features_state)

        return np.array(features)
        

    def train(self, training_samples: list['Sample'], dev_samples: list['Sample']):
        """
        Trains the MLP model using the provided training and development samples.

        This method prepares the training data by mapping samples to IDs suitable for 
        embedding layers and then proceeds to compile and fit the Keras model.

        Parameters:
            training_samples (list[Sample]): A list of training samples for the parser.
            dev_samples (list[Sample]): A list of development samples used for model validation.
        """
        # Para calcular lso vacabularios pasamos ambos conjuntos de datos (training_samples y dev_samples)
        self._feature_vocab, self._dependency_vocab = self.create_vocab(training_samples + dev_samples)
        
        # Obtenemos los datos a utilizar en el modelo
        x_train, y_train_action, y_train_dep = self.get_samples(training_samples)
        x_dev, y_dev_action, y_dev_dep = self.get_samples(dev_samples)

        # Definimos el modelo
        sample_weight_action_train = np.ones(len(training_samples), dtype=np.float32)
        sample_weight_action_dev = np.ones(len(dev_samples), dtype=np.float32)
        sample_weight_dep_train = np.ones(len(training_samples), dtype=np.float32)
        sample_weight_dep_dev = np.ones(len(dev_samples), dtype=np.float32)

        input_layer = tf.keras.layers.Input(shape=(x_train.shape[1],))

        embedding_layer = tf.keras.layers.Embedding(input_dim=len(self._feature_vocab) + 2, output_dim=self._word_emb_dim)(input_layer)
        
        flatten_layer = tf.keras.layers.Flatten()(embedding_layer)
    
        hidden = tf.keras.layers.Dense(self._hidden_dim, activation='relu')(flatten_layer)
        
        hidden = tf.keras.layers.Dense(self._hidden_dim//2, activation='relu')(hidden)
        
        transition_output = tf.keras.layers.Dense(len(np.unique(y_train_action)), activation='softmax', name="action")(hidden)

        dep_output = tf.keras.layers.Dense(len(np.unique(y_train_dep)), activation='softmax', name="dependency")(hidden)

        model = tf.keras.Model(inputs=input_layer, outputs={'action': transition_output, 'dependency': dep_output})

        # Compilamos el modelo
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss={'action': 'sparse_categorical_crossentropy', 
                            'dependency': 'sparse_categorical_crossentropy'},
                    metrics={'action': 'accuracy', 'dependency': 'accuracy'})

        if self._parada_temprana:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            callbacks = [early_stopping]
        else:
            callbacks = None

        # Entrenamos el modelo 
        history = model.fit(
            x_train,
            {'action': y_train_action, 'dependency': y_train_dep},
            sample_weight={'action': sample_weight_action_train, 'dependency': sample_weight_dep_train},
            validation_data=(x_dev, {'action': y_dev_action, 'dependency': y_dev_dep},
                            {'action': sample_weight_action_dev, 'dependency': sample_weight_dep_dev}),
            epochs=self._epochs,
            batch_size=self._batch_size,
            callbacks=callbacks)

        self.train_history = history.history['action_accuracy']
        self.dev_history = history.history['val_action_accuracy']


        # Visualizamos una gráfica con las métricas
        metrics_df = pd.DataFrame({
            'epoch': range(1, len(self.train_history) + 1),
            'train_accuracy':  self.train_history,
            'dev_accuracy': self.dev_history
        })

        plt.figure(figsize=(10, 5))
        plt.plot(metrics_df['epoch'], metrics_df['train_accuracy'], label='Train Accuracy')
        plt.plot(metrics_df['epoch'], metrics_df['dev_accuracy'], label='Dev Accuracy')
        plt.title('Evolución del Modelo')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.1) 
        plt.savefig('model_accuracy.png')
        plt.close()

        self.model = model
    
    def evaluate(self, samples: list['Sample']):
        """
        Evaluates the model's performance on a set of samples.

        This method is used to assess the accuracy of the model in predicting the correct
        transition and dependency types. The expected accuracy range is between 75% and 85%.

        Parameters:
            samples (list[Sample]): A list of samples to evaluate the model's performance.
        """
        # A partir de los samples sacamos la matriz de características 
        # y las etiquetas para las transiciones y dependencias.
        x, true_actions, true_deps = self.get_samples(samples)

        # Llamamos al modelo entrenado previamente para predecir los resultados
        # usando la matriz de características
        predictions = self.model.predict(x, batch_size=self._batch_size)
        
        action_predictions = predictions['action'] # Predicciones para las transiciones
        dep_predictions = predictions['dependency'] # Predicciones para las dependencias

        
        
        
        # Para cada predicción guardamos el índice con la mayor probabilidad
        pred_actions = np.argmax(action_predictions, axis=1)
        pred_deps = np.argmax(dep_predictions, axis=1)
        
        # Comparamos la predicción con la etiqueta real
        action_accuracy = np.mean(pred_actions == np.array(true_actions))
        dep_accuracy = np.mean(pred_deps == np.array(true_deps))
        
        # Imprimimos y devolvemos los resultados
        print(f"Action Accuracy: {action_accuracy * 100:.2f}%")
        print(f"Dependency Accuracy: {dep_accuracy * 100:.2f}%")
        
        return action_accuracy, dep_accuracy

    def run(self, sents: list[list[Token]]):
        """
        Executes the model on a list of sentences to perform dependency parsing.

        This method implements the vertical processing of sentences, predicting parser 
        transitions for each token in the sentences.

        Parameters:
            sents (list[Token]): A list of sentences, where each sentence is represented 
                                 as a list of Token objects.
        """

        # Main Steps for Processing Sentences:
        # 1. Initialize: Create the initial state for each sentence.
        # 2. Feature Representation: Convert states to their corresponding list of features.
        # 3. Model Prediction: Use the model to predict the next transition and dependency type for all current states.
        # 4. Transition Sorting: For each prediction, sort the transitions by likelihood using numpy.argsort, 
        #    and select the most likely dependency type with argmax.
        # 5. Validation Check: Verify if the selected transition is valid for each prediction. If not, select the next most likely one.
        # 6. State Update: Apply the selected actions to update all states, and create a list of new states.
        # 7. Final State Check: Remove sentences that have reached a final state.
        # 8. Iterative Process: Repeat steps 2 to 7 until all sentences have reached their final state.

        # Se crea un estado inicial para cada oración y se guarda en una lista junto a la oración correspondiente
        # Además le asignamos un índice a cada oración para mantener el orden original para despues poder comparar los trees inferidos con los reales
        current_states = [(i, arc_eager.create_initial_state(sent), sent) for i, sent in enumerate(sents)]

        # Inicializamos las variables necesarias
        final_states = []
        transition_keys = {v: k for k, v in self._transitions.items()}
        dep_keys = {v: k for k, v in self._dependency_vocab.items()}

        # Se ejecuta el bucle siempre que haya estados que no llegaran al estado final
        while current_states:
            # Separa estados que ya son finales y los que aún no lo son
            non_final = []
            final = []
            for idx, state, sent in current_states:
                if arc_eager.final_state(state):
                    # Un estado ha llegado al final cuando no quedan más tokens en 
                    # cola y todas las relaciones se han establecido
                    final.append((idx, state, sent))
                else:
                    non_final.append((idx, state, sent))
            
            # Conserva los estados que ya estaban en estado final
            final_states.extend(final)

            # Si ya no hay ninguna oración en la lista de oraciones no terminadas
            # se cierra el bucle
            if not non_final:
                break
            
            # Preparamos las muestras y extraemos las características. 
            samples = [Sample(state, None) for idx, state, sent in non_final]
            features = self.get_indices(samples)
            
            # Se hace una predicción usando el modelo entrando previamente y las características extraidas 
            predictions = self.model.predict(features, batch_size=self._batch_size)
            action_predictions = predictions['action']
            dep_predictions = predictions['dependency']

            # 4. Transition Sorting: For each prediction, sort the transitions by likelihood using numpy.argsort, 
            #    and select the most likely dependency type with argmax.
            chosen_transitions = []
            for i, (idx, state, sent) in enumerate(non_final):
                sorted_indices = np.argsort(-action_predictions[i]) # Se escojen las acciones con mayor probabilidad
                selected_action = None
                selected_dep = None
                
                # Para todas las opciones de acción comprobamos cuales son válidas,
                # escojemos la primera que sea válida (será la de mayor probabilidad)
                for idx_action in sorted_indices:
                    action = transition_keys.get(int(idx_action))
                    if action == "SHIFT":
                        valid = (len(state.B) > 0)
                    elif action == "LEFT-ARC":
                        valid = arc_eager.LA_is_valid(state)
                    elif action == "RIGHT-ARC":
                        valid = arc_eager.RA_is_valid(state)
                    elif action == "REDUCE":
                        valid = arc_eager.REDUCE_is_valid(state)
                    else:
                        valid = False
                    if valid:
                        selected_action = action
                        # Si la acción requiere una etiqueta de dependencia obtenemos la predicción correspondiente
                        if action in ["LEFT-ARC", "RIGHT-ARC"]:
                            dep_idx = np.argmax(dep_predictions[i]) # Se escoje la dependencia con mayor probabilidad 
                            selected_dep = dep_keys.get(dep_idx, "NULL")
                        break
                if selected_action is None:
                    selected_action = "SHIFT"
                # Se crea el objeto Transition incluyendo la etiqueta de dependencia si corresponde
                transition = Transition(selected_action, dependency=selected_dep)
                chosen_transitions.append(transition) 

            # Actulizamos los estados de las oraciones no finalizadas aplicando las 
            # transiciones que acabamos de seleccionar
            updated_non_final = []
            for (idx, state, sent), transition in zip(non_final, chosen_transitions):
                arc_eager.apply_transition(state, transition)
                if arc_eager.final_state(state):
                    final_states.append((idx, state, sent))
                else:
                    updated_non_final.append((idx, state, sent))
            current_states = updated_non_final # Se actualiza para que contenga solo los estados no finalizados

        # Una vez el cucle haya acabamos construimos los árboles de dependencias para cada oracion finalizada 
        # Tomamos los estados finales generados anteriormente, asignamos las realciones de dependencia
        # a cada token de la relación y contruimos los árboles sintácticos que representan la estructura 
        # gramatical de las oraciones analizadas
        final_trees = [None] * len(sents)
        for idx, state, sent in final_states:
            tree = list(sent)# convertimos la oración en una lista de tokens (para poder modificar cada token)
            # Para cada token se crean las relaciones
            # de dependencia basadas en las trasniciones guardadas en el estado
            for token in tree[1:]: # (sin contar el root)
                # Recorremos los estados de dependencia 
                for arc in state.A:
                    # Verificamos si el hijo del arco coincide con el id del token actual 
                    # para comprobar si este último tiene algún arco/dependencia ??
                    if arc[2] == token.id:
                        # Asignamos la cabeza y la etiqueta de la dependencia 
                        token.head = arc[0]
                        token.dep = arc[1]
                        break
            final_trees[idx] = tree
        final_trees = [tree for tree in final_trees if tree is not None]
        return final_trees

if __name__ == "__main__":
    model = ParserMLP()