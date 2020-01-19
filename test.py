from math import cos
from sys import stdout
from itertools import product
from random import randint
from threading import Thread

class IA(Thread):
	def __init__(self, nb_lay, deep_lay, len_outputs, cos_=False, rand=False):
		super().__init__()

		if cos_:
			self.cos = lambda x: cos(x)
		else:
			self.cos = lambda x: x

		self.nb_lay = nb_lay
		self.deep_lay = deep_lay
		self.len_outputs = len_outputs

		if rand:
			self.layers = [[randint(-20, 20)/10 for y in range(deep_lay)] for x in range(nb_lay)]
			self.outputs = [randint(-20, 20)/10 for i in range(len_outputs)]
		else:
			self.layers = [[1 for y in range(deep_lay)] for x in range(nb_lay)]
			self.outputs = [1 for i in range(len_outputs)]

		print('[Ok] Creation of the IA')

	def run_thread(self, command):
		self.command = command
		self.start()

	def run(self):
		self.command()

	def compute(self, inputs):
		from math import cos
		lay = [inputs] + self.layers + [self.outputs]

		next_ = [self.cos(sum([x*y for y in inputs])) for x in lay[1]]
		for i in range(1, len(lay)):
			next_ = [self.cos(sum([x*y for y in next_])) for x in lay[i]]
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
		
		return '\n'.join(('\nDeep learning architectur', lay, out))

	def learn_by_brute_force(self, rules, rate_similarity=0.1, deep=10, all_=False, rand=True, ptr_solution=False):
		'''
			Do not yield this function.
		'''

		print('[Ok] Starting learning process...')

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

			if sum([int(self.correspond_to(self.compute(inputs), to_get, rate_similarity)) for inputs, to_get in rules]) == len(rules):
				if ptr_solution: print(f'\n[Ok] Find a solution {str(self)}')

				solutions += [(self.layers, self.outputs)]

				if not all_:
					stdout.write('\r' + ''.join([' ' for x in range(40)])+'\n')
					stdout.flush()
					return 

			stdout.flush()

		return solutions

	def debut_code(self):
		print('==========START OF DEBUG PART==========')
		print('===========END OF DEBUG PART===========')

	def learn_experiance_once(self, rules, deeps=5, add=0.1, debug=False):

		'''
			If we want a presise learning we have to change the add factor many times.

			The deep arg correspond to (1/3|2/3), (2/3, 1/3), (1/4, 3/4), (2/4, 2/4), (3/4, 1/4), (1/5, 4/5)...

		'''

		diff = lambda lst1, lst2: sum([abs(lst1[x]-lst2[x]) for x in range(len(lst1))])

		#Make it for layers
		for deep in range(deeps):
			def apply_changes(self, x, lst, sign_lst):
				for i in range(len(lst)):
					self.layers[x][i] += sign_lst[i] * add * lst[i]/deep

			for i in range(len(self.layers)):
				for j in [k for k in product(*[range(deep) for x in range(len(self.layers[i]))]) if sum(k)==deep]:
					for signs in product(*[*[(1, -1) for x in range(len(self.layers[i]))]]):

						befor = sum([diff(rule[1], self.compute(rule[0])) for rule in rules])

						apply_changes(self, i, j, signs)

						if befor < sum([diff(rule[1], self.compute(rule[0])) for rule in rules]):
							apply_changes(self, i, j, [-x for x in signs])

			#Make it for outputs		
			def apply_changes(self, lst, sign_lst):
				for i in range(len(lst)):
					self.outputs[i] += sign_lst[i] * add * lst[i]

			for j in [k for k in product(*[range(deep) for x in range(len(self.outputs))])]:
				for signs in product(*[*[(1, -1) for x in range(len(self.outputs))]]):

					befor = sum([diff(rule[1], self.compute(rule[0])) for rule in rules])

					apply_changes(self, j, signs)

					if befor < sum([diff(rule[1], self.compute(rule[0])) for rule in rules]):
						apply_changes(self, j, [-x for x in signs])

	def learn_by_experiance_classic(self, rules, rate_similarity=0.1, cmd_gui=False):

		print('[Ok] Stating learning by experience process')

		diff = lambda lst1, lst2: sum([abs(lst1[x]-lst2[x]) for x in range(len(lst1))])

		start_from = sum([diff(rule[1], self.compute(rule[0])) for rule in rules])/len(rules)

		while sum([int(self.correspond_to(self.compute(inputs), to_get, rate_similarity)) for inputs, to_get in rules]) != len(rules):
			for i in range(-10, 10):
				self.learn_experiance_once(rules, add=i/10)

			acctualy_on = sum([diff(rule[1], self.compute(rule[0])) for rule in rules])
			
			if not cmd_gui:
				exec(f"stdout.write('\\rDiffecrance : {str(acctualy_on)}')")
			else:
				nb = int(20*acctualy_on/start_from)

				exec(f"stdout.write('\\rProgress : [{'#'*nb + ' '*(20-nb)}]')")

			stdout.flush()

	def learn_by_experiance_1(self, rules):
		'''
			The main goal of this function is to provide a fast deep learning trainner.
		'''

		print('[Ok] Stating learning by experience process 1.0')

		diff = lambda lst1, lst2: sum([abs(lst1[x]-lst2[x]) for x in range(len(lst1))])

		def test_param(link, add):
			befor = sum([diff(rule[1], self.compute(rule[0])) for rule in rules])

			while befor >= sum([diff(rule[1], self.compute(rule[0])) for rule in rules]):
				befor = sum([diff(rule[1], self.compute(rule[0])) for rule in rules])
				link += add

			link -= add

		while sum([int(self.correspond_to(self.compute(inputs), to_get, rate_similarity)) for inputs, to_get in rules]) != len(rules):
			for i in range(len(self.layers)):
				for j in range(len(self.layers[i])):
					test_param()



	def learn_by_solving(self, rules, deep=5, error_rate=0.01):

		if len(rules) == 1:
			#Step one compute the outputs param

			p = [x for x in self.outputs]

			for i in range(2, deep):
				for j in range(1, i):
					# 1/3 | 2/3
					j/i and i-j/i

if __name__ == '__main__':
	'''
		To do:
			In experiance:
				We can make 3 steps for each learn_experiance_once:
					1 - learn classic 100 times
					2 - learn in random 10 time
					3 - deeply change the ia structure 1 time (line 151 : for i in range(-10, 10) we can make (-50, -49, -48, -10, -9, -8, -7...7, 8, 9, 10, 48, 49, 50))
	'''

	from json import load
	from sys import argv
	rules = load(open(argv[1]))

	ia = IA(
		rules['numbre of layers'], 
		rules['lenght of each layers'],
		rules['lenght of outputs'],

		cos_=True,
		rand=True,
	)

	'''ia.learn_by_brute_force(
		rules['rules'],
		rate_similarity=rules['rate of similarity'],
		deep=rules['deep of presition'],
	)'''

	'''ia.learn_by_solving(
		rules['rules'],

	)'''

	'''ia.learn_experiance_once(
			rules=rules['rules'],
			add=0.1,
	)'''

	'''ia.learn_by_experiance_classic(
		rules = rules['rules'],
		rate_similarity=rules['rate of similarity'],
	)'''

	ia.learn_by_experiance_1(
		rules=rules['rules'],
	)

	print(ia)

	print('Verifying issu.')

	for inputs, outputs in rules['rules']:
		print(f'Inputs : {inputs}, outputs want : {outputs}.  Outputs get : {ia.compute(inputs)}')

	file = input('Save to (re-writing)>')

	open('file', 'w').write('\n'.join(['\n'.join(self.layers), self.outputs]))
