import tensorflow as tf

#columns def
birth_place = tf.feature_column.categorical_column_with_identity("birth_place", num_buckets=4, default_value = 0)
birth_place = tf.feature_column.indicator_column(birth_place)
#birth_place = tf.feature_column.embedding_column(birth_place, 2)

gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["male", "female"])
gender = tf.feature_column.indicator_column(gender)

department = tf.feature_column.categorical_column_with_hash_bucket("department", 4)
department = tf.feature_column.indicator_column(department)
#department = tf.feature_column.embedding_column(department, 3)

columns = [birth_place, gender, department]

#features def
#features = {
#    "birth_place":[[1],[1],[3],[4]],
#    "gender":[["male"],["female"],["male"],["female"]],
#    "department":[['sport'], ['sport'], ['drawing'], ['gardening']]
#}
features = {
    "birth_place":[1,1,3,4],
    "department":['sport', 'sport', 'drawing', 'gardening'],
    "gender":["male","female","male","female"],
}
print(features.keys())

#input layer def
input_layer = tf.keras.layers.DenseFeatures(columns)
dense_tensor = input_layer(features)
print(dense_tensor)
