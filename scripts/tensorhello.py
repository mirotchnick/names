import tensorflow as tf

graph1 = tf.Graph()

with graph1.as_default():
    a = tf.constant([2], name='constant_a')
    b = tf.constant([3], name='constant_b')

sess = tf.Session(graph = graph1)
result = sess.run(a)
sess.close()

with graph1.as_default():
    c = tf.add(a, b)

sess = tf.Session(graph = graph1)
result = sess.run(c)
result
sess.close()

with tf.Session(graph=graph1) as sess:
    result = sess.run(c)
    print(result)


graph2 = tf.Graph()
with graph2.as_default():
    Scalar = tf.constant(2)
    Vector = tf.constant([5,6,2])
    Matrix = tf.constant([[1,2,3], [2,3,4], [3,4,5]])
    Tensor = tf.constant( [ [[1,2,3], [2,3,4], [3,4,5]], [[4,5,6], [5,6,7], [6,7,8]], [[7,8,9], [8,9,10], [9,10,11]] ] )
with tf.Session(graph = graph2) as sess:
    result = sess.run(Scalar)
    print ('Scalar (1 entry):\n %s \n' % result)
    result = sess.run(Vector)
    print('Vector (3 entries):\n %s \n' % result)
    result = sess.run(Matrix)
    print('Matrix (3x3 entries):\n %s \n' % result)
    result = sess.run(Tensor)
    print('Tensor (3x3x3 entries):\n %s \n' % result)

Scalar.shape
Tensor.shape

graph3 = tf.Graph()
with graph3.as_default():
    Matrix_one = tf.constant([[1,2,3], [2,3,4], [3,4,5]])
    Matrix_two = tf.constant([[2,2,2], [2,2,2], [2,2,2]])

    add_1_operation = tf.add(Matrix_one, Matrix_two)
    add_2_operation = Matrix_one + Matrix_two

with tf.Session(graph = graph3) as sess:
    result = sess.run(add_1_operation)
    print('result from tensorflow function:')
    print(result)
    result = sess.run(add_2_operation)
    print('with base function:')
    print(result)

graph4 = tf.Graph()
with graph4.as_default():
    Mat1 = tf.constant([[1,2], [2,3]])
    Mat2 = tf.constant([[2,2], [3,3]])
    mult1 = tf.matmul(Mat1, Mat2)
    mult2 = Mat1*Mat2

with tf.Session(graph=graph4) as sess:
    result = sess.run(mult1)
    print('result from matmul:')
    print(result)


#Variables

v = tf.Variable(0)
update = tf.assign(v, v+1)
init_op = tf.global_variables_initializer()

with tf.Session() as sesh:
    sesh.run(init_op)
    print(sesh.run(v))
    for _ in range(3):
        sesh.run(update)
        print(sesh.run(v))

#Placeholders

a = tf.placeholder(tf.float32)
b = a*2

with tf.Session() as sesh:
    result = sesh.run(b, feed_dict={a:3.5})
    print(result)

dictionary = {a: [ [ [1,2,3],[4,5,6],[7,8,9],[10,11,12] ] , [ [13,14,15],[16,17,18],[19,20,21],[22,23,24] ] ] }

with tf.Session() as sesh:
    result = sesh.run(b, feed_dict=dictionary)
    print(result)


graph5 = tf.Graph()
with graph5.as_default():
    a=tf.constant([5])
    b = tf.constant([3])
    c = tf.add(a,b)
    d = tf.subtract(a,b)

with tf.Session(graph=graph5) as sesh:
    result = sesh.run(c)
    print('c =: %s' % result)
    result = sesh.run(d)
    print('d =: %s' % result)
