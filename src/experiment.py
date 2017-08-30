#!/usr/bin/env python
import time
import argparse
from kinmodel.mocap_recorder import collect_model_data
from fit_kinmodel import fit_kinmodel


class Experiment(object):

	def __init__(self, kinmodel_json):
		self.name = None
		self.age = None
		self.date = None
		self.kinmodel_json = kinmodel_json
		self.output_npz = None

	def get_details(self):
		self.name = raw_input('Enter your name or simply press ENTER to remain anonymous')
		if self.name == '':
			self.name = str(int(time.time()))
		self.name = self.name.replace(' ', '_')
		self.age = raw_input('Enter your age or simply press ENTER to remain anonymous')
		self.date = time.strftime("%d/%m/%Y")
		self.output_npz = '%s_rec.npz' % self.name

	def collect_model_data(self):
		raw_input('Stand in with your elbow by your side and your arm at ninety degrees.\n'
				  '[Peter] Press ENTER to begin collecting model data.')
		collect_model_data(self.kinmodel_json, self.output_npz)

	def optimize(self):
		fit_kinmodel(self.kinmodel_json, '%s_opt.json'%self.name, self.output_npz)

	def prepare(self):
		self.get_details()
		self.collect_model_data()
		self.optimize()

	def run(self):
		pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('kinmodel_json', help='The JSON file with the kinematic model data')
    args = parser.parse_args()
    
    try:
        experiment = Experiment(args.kinmodel_json)
        experiment.prepare()
    except rospy.ROSInterruptException:
        pass