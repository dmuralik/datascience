from __future__ import absolute_import, print_function, unicode_literals
from kafka import KafkaConsumer
from streamparse.spout import Spout
import uuid

class WordSpout(Spout):

    def initialize(self, stormconf, context):
        self.consumer = KafkaConsumer('testtopic', bootstrap_servers=['localhost:9092'])

    def next_tuple(self):
        for message in self.consumer:
            self.emit([message.value.decode('utf-8')], tup_id = str(uuid.uuid4()))




