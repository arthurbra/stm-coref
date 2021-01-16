#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import datetime

import tensorflow as tf
tf.reset_default_graph()
import util
import logging
import pyhocon
import independent
import sys

format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
  config = util.initialize_from_env()
   
  report_frequency = config["report_frequency"]
  eval_frequency = config["eval_frequency"]

  experiment_name = sys.argv[1]
  if experiment_name in ['bert_base', 'spanbert_base']:
    # calculating max_steps before changing it
    max_steps = config['num_epochs'] * config['num_docs']
    config['num_docs'] = 2802
    config['no_warmup'] = True
  else:
    max_steps = config['num_epochs'] * config['num_docs']

  model = util.get_model(config)
  saver = tf.train.Saver()

  log_dir = config["log_dir"]
  writer = tf.summary.FileWriter(log_dir, flush_secs=20)

  max_f1 = 0
  mode = 'w'

  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    model.start_enqueue_thread(session)
    accumulated_loss = 0.0

    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
      if experiment_name in ['bert_base', 'spanbert_base']:
        model.restore(session)
      else:
        print("Restoring from: {}".format(ckpt.model_checkpoint_path))
        saver.restore(session, ckpt.model_checkpoint_path)
      mode = 'a'
    
    is_first_start = not os.path.exists(os.path.join(log_dir, 'stdout.log'))
    fh = logging.FileHandler(os.path.join(log_dir, 'stdout.log'), mode=mode)
    fh.setFormatter(logging.Formatter(format))
    logger.addHandler(fh)

    config_str = '########## configuration ##########\n' + str(pyhocon.HOCONConverter.convert(config, "hocon")) + '\n##################################\n' 
    if is_first_start:
      logger.info(config_str)
    else:
      with open(os.path.join(log_dir, 'stdout.log'), mode='r') as file:
        if not 'configuration' in file.read():
          logger.info(config_str)

    initial_time, start = time.time(), time.time()
    avg_time_per_step, initial_step = 0, None
    
    while True:
      step_start_time = time.time()
      tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op])
    #   print(f'step: {tf_global_step}')

      if initial_step is None:
        initial_step = tf_global_step
      accumulated_loss += tf_loss
    
      if tf_global_step % report_frequency == 0:
        total_time = time.time() - initial_time
        steps_per_second = (tf_global_step - initial_step) / total_time
        percent_done = (tf_global_step - initial_step / max_steps) * 100

        average_loss = accumulated_loss / report_frequency
        logger.info("{:.2f}% [{}] loss={:.2f}, steps/s={:.2f}".format(percent_done, tf_global_step, average_loss, steps_per_second))
        writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
        accumulated_loss = 0.0

      if tf_global_step % eval_frequency == 0:
        saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
        # is_debugging = loads eval data from changed eval_path
        eval_summary, (p, r, eval_f1), eval_doc_keys = model.evaluate(session, tf_global_step, is_debugging=True)
        eval_path = model.config['eval_path']
        model.config['eval_path'] = model.config['train_path']
        
        _, (train_p, train_r, train_f1), train_doc_keys = model.evaluate(session, tf_global_step, is_debugging=True)
        model.config['eval_path'] = eval_path

        print('########################################################')
        print(f'## train - F1: {(train_f1 * 100):.2f}% ## P: {(train_p * 100):.2f}, R: {(train_r * 100):.2f} on {len(train_doc_keys)} docs. ##')
        print(f'## eval  - F1: {(eval_f1 * 100):.2f}% ## P: {(p * 100):.2f}, R: {(r * 100):.2f} on {len(eval_doc_keys)} docs. ##')
        print('########################################################')

        if eval_f1 > max_f1:
          max_f1 = eval_f1
          util.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))

        writer.add_summary(eval_summary, tf_global_step)
        writer.add_summary(util.make_summary({"max_eval_f1": max_f1}), tf_global_step)

        logger.info("[{}] train_f1={:.4f}, train_p={:.4f}, train_r={:.4f}".format(tf_global_step, train_f1, train_p, train_r))
        logger.info("[{}] evaL_f1={:.4f}, max_f1={:.4f}, p={:.4f}, r={:.4f}".format(tf_global_step, eval_f1, max_f1, p, r))
        
      step_elapsed_time = time.time() - step_start_time
      avg_time_per_step = (avg_time_per_step * (tf_global_step - initial_step) + step_elapsed_time) / (tf_global_step - initial_step + 1)
      if tf_global_step % report_frequency == 0:
        remaining_steps = max_steps - (tf_global_step + 1 - initial_step)
        timedelta = datetime.timedelta(seconds=(remaining_steps * avg_time_per_step))
        print(f'{percent_done:.2f}% eta: {datetime.datetime.now() + timedelta + datetime.timedelta(hours=1)} remaining: {timedelta}')
        if remaining_steps <= 0:
          print("break, tf_global_step > max_steps")
          break
