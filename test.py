class IA:
	def __init__(self, nb_lay, deep_lay, len_outputs):
		super(IA, self).__init__()

		self.nb_lay = nb_lay
		self.deep_lay = deep_lay
		self.len_outputs = len_outputs

		self.layers = [[1 for y in range(deep_lay)] for x in range(nb_lay)]
		self.outputs = [1 for i in range(len_outputs)]

		print('[Ok] Creation of the IA')

	def compute(self, inputs):
		from math import cos
		lay = [inputs] + self.layers + [self.outputs]

		next_ = [cos(sum([x*y for y in inputs])) for x in lay[1]]
		for i in range(1, len(lay)):
			next_ = [cos(sum([x*y for y in next_])) for x in lay[i]]
		return next_

	def correspond_to(self, lst1, lst2, margine):
		
		almost_same = lambda a, b, margine: abs(a-b) < margine

		for i in range(len(lst1)):
			if not almost_same(lst1[i], lst2[i], margine):
				return False
		return True

	def __str__(self):
		lay = '\nLayers:\n\t' + '\n\t'.join(f"{'  '.join([str(y) for y in line])}" for line in self.layers)
		out = f"\nOutput: {'  '.join(str(y) for y in self.outputs)}"
		
		return '\n'.join((lay, out))

	def learn(self, rules, rate_similarity=0.05, deep=10, all_=False, rand=True, ptr_solution=False):
		'''
			Do not yield this function.
		'''

		print('[Ok] Starting learning process...')

		from sys import stdout
		from itertools import product
		from random import shuffle

		solutions = []
		lp = 0

		def rnd():
			r = [x/deep for x in range(-deep, deep+1)]
			if rand:
				shuffle(r)
			return r

		for i in product(*[rnd() for j in range(len(self.outputs)+len(self.layers)*len(self.layers[0]))]): # There is 9 times
			lp += 1

			param = iter(i)

			self.layers = [[next(param) for y in range(self.deep_lay)] for x in range(self.nb_lay)]
			self.outputs = [next(param) for j in range(self.len_outputs)]

			stdout.write(f'\r[...] Stat of testing is {str(lp/21**9 * 100)[:5]}%.')

			if sum([int(self.correspond_to(self.compute(inputs), to_get, 0.1)) for inputs, to_get in rules]) == len(rules):
				if ptr_solution: print(f'\n[Ok] Find a solution {str(self)}')

				solutions += [(self.layers, self.outputs)]

				if not all_:
					stdout.write('\r' + ''.join([' ' for x in range(40)])+'\n')
					stdout.flush()
					return 

			stdout.flush()

		return solutions 

if __name__ == '__main__':
	'''
	If you want to use the rules.json , use this code
	
	from json import load
	rules = load(open('rules.json'))
	'''
	rules = {
		'rules' : [
			[[55, -16, 0.5, 1, 9], [0.8, 0.6, 0.2]],
		],

		'deep of presition' : 10,

		'numbre of layers' : 2,
		'lenght of each layers' : 2,
		'lenght of outputs' : 3,

		'rate of similarity' : 0.01,
	}

	ia = IA(
		rules['numbre of layers'], 
		rules['lenght of each layers'],
		rules['lenght of outputs'],
	)

	'''result = ia.test(
		rules['rules'][0][0], rules['rules'][0][1], 
		rules['deep of presition'], 
		rules['rate of similarity']
	)'''

	ia.learn(
		rules['rules'],
		rate_similarity=rules['rate of similarity'],
		deep=rules['deep of presition'],
	)

	print(ia)
