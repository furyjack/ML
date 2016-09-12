import tensorflow as tf
x1=5
x2=6
sess=tf.Session()
result=tf.mul(x1,x2)
print sess.run(result)
sess.close()