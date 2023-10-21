import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline


def split_data(df):
    features = list(df.columns)[:-1]
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    x_train = x.iloc[:-30, :]
    y_train = y.iloc[:-30]

    x_test = x.iloc[-30:, :]
    y_test = y.iloc[-30:]

    return x_train, y_train, x_test, y_test, features


def scale_data(x_train, x_test, features):
    scaler = MinMaxScaler()

    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train))
    x_test_scaled = pd.DataFrame(scaler.transform(x_test))
    x_train_scaled.columns = features
    x_test_scaled.columns = features

    return x_train_scaled, x_test_scaled


def mlp_classifier(x_train, y_train, x_test, y_test):
    model = tf.keras.Sequential(
        [
            layers.BatchNormalization(),
            layers.Dense(units=16, activation="sigmoid"),
            # layers.BatchNormalization(),
            layers.Dense(units=1, activation="sigmoid"),
        ]
    )

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss=loss_fn,
        metrics="binary_accuracy",
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=500,
        batch_size=128,
        validation_data=(x_test, y_test),
    )
    # print(model.weights)
    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ["loss", "val_loss"]].plot()
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.figure()
    history_df.loc[:, ["binary_accuracy", "val_binary_accuracy"]].plot()
    plt.title("Binary accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Accuracy")


def logistic_model(x_train, y_train, x_test, y_test):
    lm = LogisticRegression().fit(x_train, y_train)
    print("Training score")
    print(lm.score(x_train, y_train))
    print("Test score")
    print(lm.score(x_test, y_test))
    print(f"linear model coefficients: ")
    print(lm.coef_)


def decision_tree(x_train, y_train, x_test, y_test, features):
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    print("Test score")
    print(clf.score(x_test, y_test))
    tree.plot_tree(clf, feature_names=features, fontsize=10)


def sklearn_mlp_classifier(x_train, y_train, x_test, y_test):
    clf = MLPClassifier(
        hidden_layer_sizes=[16, 1], activation="logistic", solver="adam", max_iter=1500
    )

    clf.fit(x_train, y_train)
    print("Test score")
    print(clf.score(x_test, y_test))


def random_forest(x_train, y_train, x_test, y_test):
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(x_train, y_train)
    print("Test score")
    print(clf.score(x_test, y_test))


def svc(x_train, y_train, x_test, y_test):  # Support vector classification
    clf = make_pipeline(StandardScaler(), SVC())
    clf.fit(x_train, y_train)

    print("Test score")
    print(clf.score(x_test, y_test))
