#!/bin/bash
rm -f protos/agents/*_pb2.py
rm -f protos/agents/base/*_pb2.py
rm -f protos/approximators/*_pb2.py
rm -f protos/approximators/base/*_pb2.py
rm -f protos/models/*_pb2.py
rm -f protos/models/base/*_pb2.py
rm -f protos/*_pb2.py
rm -f protos/elements/*_pb2.py
rm -f protos/buffers/*_pb2.py
rm -f protos/*_pb2.py

rm -rf protos/agents/__pycache__
rm -rf protos/agents/base/__pycache__
rm -rf protos/approximators/__pycache__
rm -rf protos/approximators/base/__pycache__
rm -rf protos/models/__pycache__
rm -rf protos/models/base/__pycache__
rm -rf protos/buffers/__pycache__
rm -rf protos/__pycache__
rm -rf protos/elements/__pycache__
rm -rf protos/__pycache__
