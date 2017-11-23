#!/usr/bin/env python
import time
import os
import argparse
from kinmodel.mocap_recorder import collect_model_data
from fit_kinmodel import fit_kinmodel


HOME = os.path.expanduser('~')


class Experiment(object):
    base_path = HOME + '/experiment/'

    def __init__(self, kinmodel_json):
        self.name = None
        self.age = None
        self.date = None
        self.kinmodel_json = kinmodel_json
        self.output_npz = None
        self.output_json = None
        self.directory = None

    def get_details(self):
        self.name = raw_input('Enter your name or simply press ENTER to remain anonymous:\n')
        if self.name == '':
            self.name = str(int(time.time()))
        self.name = self.name.replace(' ', '_')
        self.age = raw_input('Enter your age or simply press ENTER to remain anonymous:\n')
        self.date = time.strftime("%d/%m/%Y")
        self.directory = self.base_path + self.name
        self.output_npz = '%s/%s_rec.npz' % (self.directory, self.name)
        self.output_json = '%s/%s_opt.json' % (self.directory, self.name)

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        details = open('%s/%s.txt' % (self.directory, self.name), 'w')
        details.write('Age: %s\nDate: %s\n' % (self.age, self.date))
        details.close()

    def collect_model_data(self):
        collect_model_data(self.kinmodel_json, self.output_npz, timed_counts=-1)

    def optimize(self):
        fit_kinmodel(self.kinmodel_json, self.output_json, self.output_npz)

    def prepare(self, learn=True):
        self.get_details()
        self.collect_model_data()
        if learn:
            self.optimize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kinmodel_json', help='The JSON file with the kinematic model data')
    args = parser.parse_args()
    experiment = Experiment(args.kinmodel_json)
    experiment.prepare(False)
