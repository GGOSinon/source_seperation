def make_grad(s, e, name):
	var = tf.get_variable(name, [3, 3, s, e], initializer=tf.contrib.layers.xavier_initializer())
	var_list.append(var)
	return var

def make_bias(x, name):
	var = tf.get_variable(name, [x], initializer=tf.contrib.layers.xavier_initializer())
	var_list.append(var)
	return var

def make_dict(num_layer, num_filter, end_str, s_filter = 1, e_filter = 1):							     
	result = {}
	weights = {}
	biases = {}

	weights['wc1'] = make_grad(s_filter,num_filter,"w1"+end_str)
	for i in range(2, num_layer):
		index = 'wc' + str(i)
		name = 'w' + str(i) + end_str
		weights[index] = make_grad(num_filter, num_filter, name)
	weights['wcx'] = make_grad(num_filter,e_filter,"wx"+end_str)
	
	biases['bc1'] = make_bias(num_filter,"b1"+end_str)
	for i in range(2, num_layer):
		index = 'bc' + str(i)
		#print(index)
		name = 'b' + str(i) + end_str
		biases[index] = make_bias(num_filter, name)
	biases['bcx'] = make_bias(e_filter,"bx"+end_str)

	result['weights'] = weights
	result['biases'] = biases
	return result
