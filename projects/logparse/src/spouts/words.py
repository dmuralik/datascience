from __future__ import absolute_import, print_function, unicode_literals

from streamparse.spout import Spout
import uuid

class WordSpout(Spout):

    def initialize(self, stormconf, context):
        self.readFile = 'data/input.log'
        self.completed = False

    def next_tuple(self):
        if (self.completed):
            return
        for line in open(self.readFile):
            self.emit([line], tup_id = str(uuid.uuid4()))
        self.completed = True



