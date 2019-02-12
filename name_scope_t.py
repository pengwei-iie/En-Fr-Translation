# coding: utf-8
import tensorflow as tf
# https://blog.csdn.net/lucky7213/article/details/78967306

# ************************************************** 讲述name_scope和variable_scope对tf.get_variable的影响 #
with tf.name_scope("a_name_scope") as myscope:
    initializer = tf.constant_initializer(value=1)
    var1_ = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1_.name)        # var1:0
    print(sess.run(var1_))   # [ 1.]

# with tf.name_scope("a_name_scope_new") as myscope:
#     initializer = tf.constant_initializer(value=1)
#     var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
# 上面这个变量名var1，写成var2也不行 说明name='var1'已经见过了（即使name_scope("a_name_scope_new")是新的名字）
# ValueError: Variable var1 already exists, disallowed.

# with tf.name_scope("a_name_scope_new", reuse = True) as myscope:
#     initializer = tf.constant_initializer(value=1)
#     var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
# no reuse arrtribute

print()
with tf.name_scope("a_name_scope") as myscope:
    #initializer = tf.constant_initializer(value=1)
    var2_ = tf.get_variable(name='var2', dtype=tf.float32, initializer=var1_)
    #var2=var1
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1_.name)        # var1:0
    print(sess.run(var1_))   # [ 1.]
    print(var2_.name)        # var2:0
    print(sess.run(var2_))   # [ 1.]
# assert var1 == var2 # error 地址不一样  值是一样的
#
print()
with tf.name_scope("a_name_scope") as myscope:
    var3_=var1_ # 直接把地址赋值
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1_.name)        # var1:0
    print(sess.run(var1_))   # [ 1.]
    print(var3_.name)        # var1:0
    print(sess.run(var3_))   # [ 1.]

assert var1_ == var3_
#
# ************************************************** 此处的var1和之前的是不等的#
print()
with tf.variable_scope("a_variable_scope") as myscope:
    initializer = tf.constant_initializer(value=3)
    v1_ = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(v1_.name)        # a_variable_scope/var1:0
    print(sess.run(v1_))   # [ 3.]

print()
# name='var1' 是关键，只要不重用，不管前面的变量名叫1还是叫2都没用
# name='var2' 即使重用了也报错，因为var2在前面tf.name_scope("a_name_scope") 已经定义了
# ValueError: Variable a_variable_scope/var1 already exists, disallowed.
with tf.variable_scope("a_variable_scope", reuse=True) as myscope:
    initializer = tf.constant_initializer(value=2)
    var2 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
    # var2= v1_
    # 效果一样,现在var2不再是前面tf.name_scope("a_name_scope")，而变成了a_variable_scope/var1:0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var2.name)        # a_variable_scope/var1:0
    print(sess.run(var2))   # [ 3.]

print()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(v1_.name)        # a_variable_scope/var1:0
    print(sess.run(v1_))   # [ 3.]
    print(var2.name)        # a_variable_scope/var1:0
    print(sess.run(var2))   # [ 3.]
    print(var3_.name)        # var1:0
    print(sess.run(var3_))   # [ 1.]
#
'''
下面会报错，因为‘var2’这个变量没有创建，需要去掉myscope.reuse_variables()这句话
print()
with tf.variable_scope("a_variable_scope") as myscope:
    initializer = tf.constant_initializer(value=1)
    myscope.reuse_variables()
    v2_ = tf.get_variable(name='var2', shape=[1], dtype=tf.float32, initializer=initializer)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(v1_.name)        # a_variable_scope/var1:0
    print(sess.run(v1_))   # [ 3.]
    print(v2_.name)        # a_variable_scope/var1:0
    print(sess.run(v2_))   # [ 3.]
'''

print('...')
with tf.variable_scope("a_variable_scope") as myscope:
    initializer = tf.constant_initializer(value=5)
    # myscope.reuse_variables() 因为var2没有创建
    var2 = tf.get_variable(name='var2', shape=[1], dtype=tf.float32, initializer=initializer)
    var7= tf.get_variable(name='var7', shape=[1], dtype=tf.float32, initializer=initializer)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(v1_.name)        # a_variable_scope/var1:0
    print(sess.run(v1_))   # [ 1.]
    print(var2.name)        # a_variable_scope/var1:0
    print(sess.run(var2))   # [ 1.]
    print(var7.name)        # a_variable_scope/var1:0
    print(sess.run(var7))   # [ 1.]


# with tf.name_scope("a_name_scope") as myscope:
#     var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(v1_.name)        # a_variable_scope/var1:0
#     print(sess.run(v1_))   # [3.]
#     print(var2.name)        # a_name_scope/var2:0
#     print(sess.run(var2))   # [ 2.]
#
#
# print('4')
# with tf.name_scope("a_name_scope") as myscope:
#     var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(v1_.name)        # a_variable_scope/var1:0
#     print(sess.run(v1_))   # [3.]
#     print(var2.name)        # a_name_scope/var2:0
#     print(sess.run(var2))   # [ 2.]