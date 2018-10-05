#!/bin/bash
rm protos/agents/*_pb2.py
rm protos/approximators/*_pb2.py
rm protos/models/*_pb2.py
rm protos/*_pb2.py
rm protos/elements/*_pb2.py
rm protos/buffers/*_pb2.py

rm -r protos/agents/__pycache__
rm -r protos/approximators/__pycache__
rm -r protos/models/__pycache__
rm -r protos/buffers/__pycache__
rm -r protos/__pycache__
rm -r protos/elements/__pycache__
