#!/bin/bash
rm -f advantage/protos/agents/*_pb2.py
rm -f advantage/protos/agents/base/*_pb2.py
rm -f advantage/protos/approximators/*_pb2.py
rm -f advantage/protos/approximators/base/*_pb2.py
rm -f advantage/protos/models/*_pb2.py
rm -f advantage/protos/models/base/*_pb2.py
rm -f advantage/protos/*_pb2.py
rm -f advantage/protos/elements/*_pb2.py
rm -f advantage/protos/buffers/*_pb2.py
rm -f advantage/protos/*_pb2.py

rm -rf advantage/protos/agents/__pycache__
rm -rf advantage/protos/agents/base/__pycache__
rm -rf advantage/protos/approximators/__pycache__
rm -rf advantage/protos/approximators/base/__pycache__
rm -rf advantage/protos/models/__pycache__
rm -rf advantage/protos/models/base/__pycache__
rm -rf advantage/protos/buffers/__pycache__
rm -rf advantage/protos/__pycache__
rm -rf advantage/protos/elements/__pycache__
rm -rf advantage/protos/__pycache__
