import mysql.connector
import datetime

import support


def insertDatabase(query, args):
    try:
        mydb = support.configureDatabase()

        mydb.set_converter_class(NumpyMySQLConverter)
        cursor = mydb.cursor()
        cursor.execute(query, args)

        if cursor.lastrowid:
            print('last insert id', cursor.lastrowid)
        else:
            print('last insert id not found')

        mydb.commit()
    except mysql.connector.Error as error:
        print(error)

    return cursor.lastrowid

def getTraining():
    ans = None
    try:
        mydb = support.configureDatabase()

        cursor = mydb.cursor()
        cursor.execute("SELECT * FROM rna_trainings WHERE is_active = 1")
        ans = cursor.fetchall()

        return ans

    except mysql.connector.Error as error:
        print(error)

    return ans

def getFirstHiddenWeights():
    training = getTraining()
    if training.__len__() > 0:
        mydb = support.configureDatabase();

        cursor = mydb.cursor()
        cursor.execute("SELECT neuron1, neuron2, neuron3, neuron4 FROM first_hidden_weights WHERE train_id = %s", (training[0][0],))

        ans = cursor.fetchall()

        return ans

def getSecondHiddenWeights():
    training = getTraining()
    if training.__len__() > 0:
        mydb = support.configureDatabase();

        cursor = mydb.cursor()
        cursor.execute("SELECT neuron1, neuron2 FROM second_hidden_weights WHERE train_id = %s", (training[0][0],))

        ans = cursor.fetchall()

        return ans

def getOutputWeights():
    training = getTraining()
    if training.__len__() > 0:
        mydb = support.configureDatabase();

        cursor = mydb.cursor()
        cursor.execute("SELECT neuron1 FROM output_weights WHERE train_id = %s", (training[0][0],))

        ans = cursor.fetchall()

        return ans

def insertTraining(first_hidden_weights, second_hidden_weights, output_weights, ni, nh1, nh2,  no):
    training = getTraining()
    if training.__len__() > 0:
        mydb = support.configureDatabase()

        cursor = mydb.cursor()
        cursor.execute("UPDATE rna_trainings SET is_active = %s WHERE id = %s", (False, training[0][0]))
        mydb.commit()

    query = "INSERT INTO rna_trainings(train_date, user_id, is_active) " \
            "VALUES(%s, %s, %s)"
    args = (datetime.datetime.now(), 1, True)
    train_id = insertDatabase(query, args)

    for i in range(ni):
        query = "INSERT INTO first_hidden_weights(neuron1, neuron2, neuron3, neuron4, train_id) " \
                "VALUES(%s, %s, %s, %s, %s)"
        args = (first_hidden_weights[i][nh1 - 4],
                first_hidden_weights[i][nh1 - 3],
                first_hidden_weights[i][nh1 - 2],
                first_hidden_weights[i][nh1 - 1],
                train_id)
        hidden_weight_id = insertDatabase(query, args)

    for j in range(nh1):
        query = "INSERT INTO second_hidden_weights(neuron1, neuron2, train_id) " \
                "VALUES(%s, %s, %s)"
        args = (second_hidden_weights[j][nh2 - 2],
                second_hidden_weights[j][nh2 - 1],
                train_id)
        hidden_weight_id = insertDatabase(query, args)

    for k in range(nh2):
        query = "INSERT INTO output_weights(neuron1, train_id) " \
                "VALUES(%s, %s)"
        args = (output_weights[k][no - 1],
                train_id)
        output_weight_id = insertDatabase(query, args)


class NumpyMySQLConverter(mysql.connector.conversion.MySQLConverter):
    """ A mysql.connector Converter that handles Numpy types """

    def _float32_to_mysql(self, value):
        return float(value)

    def _float64_to_mysql(self, value):
        return float(value)

    def _int32_to_mysql(self, value):
        return int(value)

    def _int64_to_mysql(self, value):
        return int(value)

