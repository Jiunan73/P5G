import os
import logging
import time
from datetime import datetime
from typing import Iterable
from queue import Queue
from threading import Thread
from concurrent.futures.thread import ThreadPoolExecutor
from base64 import b64encode
from flask import (Flask, jsonify, render_template, request, Response)
import cv2
import numpy as np
import requests
from camera.webcam import Webcam
from camera.onvif_camera import clsONVIFCamera
import IR_cam.ImageFormatControl_QuickSpin as IRCam
from IR_cam.FLIR_A400 import FLIRA400
from Sockets.ftp_utils import MyFTP
from Sockets.ADS import ADSClient
from file_utils.operate_file import (write_file, read_lines)
from config_utils.config_utils import ClsConfigParser
from DB.clsMySqlDB import clsMySqlDB
from Logger.LogWriter import LogWriter
from panorama.panorama import (stitch, crop, add_black_margin)


class AMR(ADSClient):

    def __init__(self, ams_net_id: str = None):
        super().__init__(ams_net_id)

        self.amr_pos_x = None
        self.amr_pos_y = None
        self.amr_pos_z = None
        self.amr_pos_theta = None
        self.amr_tag_id = None

        self.camera = None
        self.logger = LogWriter("AMR")
        self.heart_bit = 0
        self.initialize_state = False # initialize state
        
        # web control
        self.ManualSemiControlDisableStatus = None # 0: enable; 1:disable
        self.AutoManualStatus = None # 0:manual; 1:auto
        self.AutoManualSwitch = None # swtitch 1 and then wait until AutoManualStatus = 0
        self.web_control_counter = 0
        self.web_control_thread = Thread(target=self.web_control_heartbeat) # web control heart beat
        self.web_control_thread.start()
        self.stop_thread = Thread()
        self.LastRealSenseObstacleSignal = None
        self.RealSenseObstacleSignal = None 
        self.obstacle_thread = Thread() # obstacle thread

    def do_routine_job(self, client):
        """??AMR??"""

        success = True 

        # check rtsp
        if not self.camera.is_running(): # ptz camera
            print("camera is not running , restart camera!")
            self.logger.error("camera is not running , restart camera!")
            self.camera.close_camera()
            self.camera.open_camera(True)
            time.sleep(1)
        if not self.camera.front_camera.is_running(): # front camera
            print("front camera is not running , restart front camera!")
            self.logger.error("front camera is not running , restart camera!")
            self.camera.front_camera.close_camera()
            self.camera.front_camera.open_camera(True)
            time.sleep(1)
        if not self.camera.ir_cam.is_running: # ir cam
            self.camera.ir_cam.close_camera()
            self.camera.ir_cam = FLIRA400()  
            self.camera.ir_cam.open_camera()
            time.sleep(1)    

        # check mysql conn
        if not self.camera.mysql_conn.IsOpen():
            print("mysql is not connected, restart mysql connection")
            self.logger.error("mysql is not connected, reconnect to mysql")
            self.camera.mysql_conn.Close()
            self.camera.mysql_conn.Open()
        
        # get PTZ status
        ptz_status = self.camera.onvif.get_ptz_status()
        pan, tilt, zoom = ptz_status.get("pan"), \
            ptz_status.get("tilt"), \
            ptz_status.get("zoom")
        if isinstance(pan, float) and isinstance(tilt, float) and isinstance(zoom, float):
            self.camera.crr_pan, self.camera.crr_tilt, self.camera.crr_zoom = calculate_ptz_angle([pan, tilt, zoom])
        
        # amr heart bit
        try:
            client.write_by_name("GVL.ExternalDevice1.bFromExternalDeviceHartBit", self.heart_bit)
        except:
            self.logger.error("write heart bit failed!")
            success = False
            return success
        finally:
            if self.heart_bit == 1:
                self.heart_bit = 0
            else:
                self.heart_bit = 1

        # get amr status (ready function)
        try:
            pos_x = client.read_by_name("GVL.nCar_PositionX")
            pos_y = client.read_by_name("GVL.nCar_PositionY")
            pos_z = client.read_by_name("GVL.nCarLiftHeight")
            pos_theta = client.read_by_name("GVL.nCar_PositionYaw")
            tag_id = client.read_by_name("GVL.nCar_PositionTagID")
            CameraWorkStatus = client.read_by_name("GVL.CameraWorkStatus") # camera status
            ToCameraWorkCommand = client.read_by_name("GVL.ToCameraWorkCommand") 
            FromCameraWorkCommand = client.read_by_name("GVL.FromCameraWorkCommand")
            # amr manual control
            ManualSemiControlDisableStatus = client.read_by_name("GVL.bWeb_ManualSemiControlDisableStatus")
            AutoManualStatus = client.read_by_name("GVL.bWeb_AutoManualStatus")
            AutoManualSwitch = client.read_by_name("GVL.bWeb_AutoManualSwitch")
        except Exception as e:
            self.logger.error(f"get amr ready function failed, error:{e}")
            success = False
            return success
        else:
            # update amr information
            self.amr_pos_x = pos_x  
            self.amr_pos_y = pos_y
            self.amr_pos_z = pos_z
            self.amr_tag_id = tag_id
            self.amr_pos_theta = pos_theta // 1000 # get integer
            # amr manual control panameter
            self.ManualSemiControlDisableStatus = ManualSemiControlDisableStatus
            self.AutoManualStatus = AutoManualStatus
            self.AutoManualSwitch = AutoManualSwitch
        
        # get amr status (new function)
        try:
            # amr obstacle detection
            RealSenseObstacleSignal = client.read_by_name("GVL.bWeb_RealSenseObstacleSignal")
        except Exception as e:
            self.logger.error(f"get amr new function failed, error:{e}")
        else:
            # update amr information
            self.RealSenseObstacleSignal = RealSenseObstacleSignal

        # report status
        self.logger.info(f"heart bit:{self.heart_bit}")
        self.logger.info(f"FromCameraWorkCommand:{FromCameraWorkCommand}, ToCameraWorkCommand:{ToCameraWorkCommand}, CameraWorkStatus:{CameraWorkStatus}")    
        self.logger.info(f"pan:{self.camera.crr_pan}, tilt:{self.camera.crr_tilt}, zoom:{self.camera.crr_zoom}")
        self.logger.info(f"pos_x:{self.amr_pos_x}, pos_y:{self.amr_pos_y}, pos_theta:{self.amr_pos_theta}, tag_id:{tag_id}, pos_z:{pos_z}")
        self.logger.info(f"Max Temperature:{camera.ir_cam.max_temperature_float}, max_coords:{camera.ir_cam.max_coords}")
        self.logger.info(f"ManualSemiControlDisableStatus:{self.ManualSemiControlDisableStatus}, AutoManualStatus:{self.AutoManualStatus}, AutoManualSwitch:{self.AutoManualSwitch}")
        # report task status
        if self.camera.target_task.is_running:
            self.logger.info(f"target_task:{self.camera.target_task.is_running}")
        if self.camera.ir_task.is_running:
            self.logger.info(f"ir_task:{self.camera.ir_task.is_running}")  
        if self.camera.designated_task.is_running:
            self.logger.info(f"designated_task:{self.camera.designated_task.is_running}")  
        if self.camera.initial_task.is_running:
            self.logger.info(f"initial_task:{self.camera.initial_task.is_running}")
        if self.camera.video_task.is_running:
            self.logger.info(f"video_task:{self.camera.video_task.is_running}") 

        # report new function status
        self.logger.info(f"RealSenseObstacleSignal:{self.RealSenseObstacleSignal}, LastRealSenseObstacleSignal:{self.LastRealSenseObstacleSignal}")

        # obstacle detection
        if self.RealSenseObstacleSignal == True and self.LastRealSenseObstacleSignal == False:
            # download and upload obstacle imgs
            self.obstacle_thread = Thread(target=self.download_and_upload_obstacle_imgs)
            self.obstacle_thread.start() 

        # tasks
        if ToCameraWorkCommand == 7:  # designated task

            if CameraWorkStatus == 0: 

                if FromCameraWorkCommand == 0:

                    ################ start designated task ################
                    self.logger.debug("start designated task, write FromCameraWorkCommand = 7, and write CameraWorkStatus = 1")
                    
                    camera.designated_task.is_running = True # change status

                    url = "http://127.0.0.1:8080/designated/start_designated_task"
                    ret, response = self.do_tasks(url)
                    self.logger.debug(f"request result----ret:{ret}, response:{response}")
                    if ret and isinstance(response, Iterable):
                        if "status" in response and "message" in response:
                            if response.get("status"):
                                
                                # PLC OK
                                try:
                                    client.write_by_name("GVL.CameraWorkStatus", 1)
                                    client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                                except:
                                    success = False
                                    return success
                                else:
                                    self.logger.debug("start designated task successfully!")
                            else:

                                # status failed
                                self.logger.error("status failed, write CameraWorkStatus = 2, FromCameraCommand = 7")

                                # PLC NG
                                try:
                                    client.write_by_name("GVL.CameraWorkStatus", 2)  
                                    client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                                except:
                                    success = False
                                    return success
                        else:

                            # type failed
                            self.logger.error("type failed, write CameraWorkStatus = 2, FromCameraCommand = 7")

                            # PLC NG
                            try:
                                client.write_by_name("GVL.CameraWorkStatus", 2)
                                client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                            except:
                                success = False
                                return success
                    else:
                        # request failed
                        self.logger.error("request failed, write CameraWorkStatus = 2, FromCameraCommand = 7")

                        # PLC NG
                        try:
                            client.write_by_name("GVL.CameraWorkStatus", 2)
                            client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                        except:
                            success = False
                            return success
                    ################ start designated task ################
            
            elif CameraWorkStatus == 1:

                ################ wait designated task complete ################
                if not self.camera.designated_task.is_running:
                    # task completed!
                    self.logger.debug("designated task completed, write CameraWorkStatus = 0")
                    # PLC D???
                    try:
                        client.write_by_name("GVL.CameraWorkStatus", 0)  
                    except:
                        success = False
                        return success
                else:
                    self.logger.debug("waiting for designated task completed!")
                ################ wait designated task complete ################

        elif ToCameraWorkCommand == 8: # IR camera
            
            if CameraWorkStatus == 0: 

                if FromCameraWorkCommand == 0:

                    ################ start ir task ################
                    self.logger.debug("start ir task, write FromCameraWorkCommand = 8, and write CameraWorkStatus = 1")

                    camera.ir_task.is_running = True # change status

                    url = "http://127.0.0.1:8080/ir/start_ir_task"
                    ret, response = self.do_tasks(url)
                    self.logger.debug(f"request result----ret:{ret}, response:{response}")
                    if ret and isinstance(response, Iterable):
                        if "status" in response and "message" in response:
                            if response.get("status"):
                                
                                # PLC ok
                                try:
                                    client.write_by_name("GVL.CameraWorkStatus", 1)  # CameraWorkStatus????1,?????
                                    client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                                except:
                                    success = False
                                    return success
                                else:
                                    self.logger.debug("start ir task successfully!")
                            else:

                                # status failed
                                self.logger.error("status failed, write CameraWorkStatus = 2, FromCameraCommand = 8")

                                # PLC NG
                                try:
                                    client.write_by_name("GVL.CameraWorkStatus", 2)  # CameraWorkStatus????1,?????
                                    client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                                except:
                                    success = False
                                    return success
                        else:

                            # type failed
                            self.logger.error("type failed, write CameraWorkStatus = 2, FromCameraCommand = 8")

                            # PLC NG
                            try:
                                client.write_by_name("GVL.CameraWorkStatus", 2)  # CameraWorkStatus????1,?????
                                client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                            except:
                                success = False
                                return success
                    else:
                        # request failed
                        self.logger.error("request failed, write CameraWorkStatus = 2, FromCameraCommand = 8")

                        # PLC NG
                        try:
                            client.write_by_name("GVL.CameraWorkStatus", 2)  # CameraWorkStatus????1,?????
                            client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                        except:
                            success = False
                            return success
                    ################ start ir task ################
            
            elif CameraWorkStatus == 1:

                ################ wait ir task complete ################
                if not self.camera.ir_task.is_running:
                    # task completed!
                    self.logger.debug("ir task completed, write CameraWorkStatus = 0")
                    # PLC D???
                    try:
                        client.write_by_name("GVL.CameraWorkStatus", 0)  
                    except:
                        success = False
                        return success
                else:
                    self.logger.debug("waiting for ir task completed!")
                ################ wait ir task complete ################
        
        elif ToCameraWorkCommand == 9: # target task

            if CameraWorkStatus == 0: 

                if FromCameraWorkCommand == 0:
                  
                    ################ start target task ################
                    self.logger.debug("start target task, write FromCameraWorkCommand = 9, and write CameraWorkStatus = 1")

                    camera.target_task.is_running = True # change status
                    url = "http://127.0.0.1:8080/target/start_target_task"
                    ret, response = self.do_tasks(url)
                    self.logger.debug(f"request result----ret:{ret}, response:{response}")
                    if ret and isinstance(response, Iterable):
                        if "status" in response and "message" in response:
                            if response.get("status"):
                                # PLC OK
                                try:
                                    client.write_by_name("GVL.CameraWorkStatus", 1)  # CameraWorkStatus 1
                                    client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                                except:
                                    success = False
                                    return success
                                else:
                                    self.logger.debug("start target task successfully!")
                            else:
                                # status failed
                                self.logger.error("status failed, write CameraWorkStatus = 2, FromCameraCommand = 9")

                                # PLC NG
                                try:
                                    client.write_by_name("GVL.CameraWorkStatus", 2) 
                                    client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                                except:
                                    success = False
                                    return success
                        else:

                            # type failed
                            self.logger.error("type failed, write CameraWorkStatus = 2, FromCameraCommand = 9")

                            # PLC NG
                            try:
                                client.write_by_name("GVL.CameraWorkStatus", 2)  # CameraWorkStatus????1,?????
                                client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                            except:
                                success = False
                                return success
                    else:
                        # request failed
                        self.logger.error("request failed, write CameraWorkStatus = 2, FromCameraCommand = 9")

                        # PLC NG
                        try:
                            client.write_by_name("GVL.CameraWorkStatus", 2)  # CameraWorkStatus????1,?????
                            client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                        except:
                            success = False
                            return success  
                    ################ start target task ################
            
            elif CameraWorkStatus == 1: #camera working
                
                ################ wait target task complete ################
                if not self.camera.target_task.is_running:
                    # task completed!
                    self.logger.debug("target task completed, write CameraWorkStatus = 0")
                    try:
                        client.write_by_name("GVL.CameraWorkStatus", 0)  # CameraWorkStatus 0, complted
                        # client.write_by_name("GVL.FromCameraWorkCommand", 0)
                    except:
                        success = False
                        return success
                    else:
                        # self.initialize_state = False # initialize state
                        pass
                else:
                    self.logger.debug("waiting for target task completed!")
                ################ wait target task complete ################         
                    
        elif ToCameraWorkCommand == 10: # panorama task 

            if CameraWorkStatus == 0: 

                if FromCameraWorkCommand == 0:
                
                    ################ start panorama task ################
                    self.logger.debug("start panorama task, write FromCameraWorkCommand = 10, and write CameraWorkStatus = 1")

                    camera.panorama_task.is_running = True # change status

                    url = "http://127.0.0.1:8080/panorama/start_panorama_task"
                    ret, response = self.do_tasks(url)
                    self.logger.debug(f"request result----ret:{ret}, response:{response}")
                    if ret and isinstance(response, Iterable):
                        if "status" in response and "message" in response:
                            if response.get("status"):
                                
                                # PLC D???
                                try:
                                    client.write_by_name("GVL.CameraWorkStatus", 1)  # CameraWorkStatus????1,?????
                                    client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                                except:
                                    success = False
                                    return success
                                else:
                                    self.logger.debug("start panorama task successfully!")
                            else:

                                # status failed
                                self.logger.error("status failed, write CameraWorkStatus = 2, FromCameraCommand = 10")

                                # PLC D??? NG
                                try:
                                    client.write_by_name("GVL.CameraWorkStatus", 2)  # CameraWorkStatus????1,?????
                                    client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                                except:
                                    success = False
                                    return success
                        else:

                            # type failed
                            self.logger.error("type failed, write CameraWorkStatus = 2, FromCameraCommand = 10")

                            # PLC NG
                            try:
                                client.write_by_name("GVL.CameraWorkStatus", 2)  # CameraWorkStatus????1,?????
                                client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                            except:
                                success = False
                                return success
                    else:
                        # request failed
                        self.logger.error("request failed, write CameraWorkStatus = 2, FromCameraCommand = 10")

                        # PLC D??? NG
                        try:
                            client.write_by_name("GVL.CameraWorkStatus", 2)  # CameraWorkStatus????1,?????
                            client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                        except:
                            success = False
                            return success
                    ################ start panorama task ################
            
            elif CameraWorkStatus == 1:

                ################ wait panorama task complete ################
                if not self.camera.panorama_task.is_running:
                    # task completed!
                    self.logger.debug("panorama task completed, write CameraWorkStatus = 0")
                    # PLC D
                    try:
                        client.write_by_name("GVL.CameraWorkStatus", 0)
                    except:
                        success = False
                        return success
                else:
                    self.logger.debug("waiting for panoramra task completed!")
                ################ wait panorama task complete ################
        
        elif ToCameraWorkCommand == 11: # video task    
            
            if CameraWorkStatus == 0: 

                if FromCameraWorkCommand == 0:

                    ################ start video task ################
                    self.logger.debug(f"start video task, write FromCameraWorkCommand = {ToCameraWorkCommand}, and write CameraWorkStatus = 1")

                    camera.video_task.is_running = True # change status
                    url = "http://127.0.0.1:8080/video/start_video_task"
                    ret, response = self.do_tasks(url)
                    self.logger.debug(f"request result----ret:{ret}, response:{response}")
                    if ret and isinstance(response, Iterable):
                        if "status" in response and "message" in response:
                            if response.get("status"):
                                # PLC OK
                                try:
                                    client.write_by_name("GVL.CameraWorkStatus", 1) 
                                    client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                                except:
                                    success = False
                                    return success
                                else:
                                    self.logger.debug("start video task successfully!")
                            else:
                                # status failed
                                self.logger.error(f"status failed, write CameraWorkStatus = 2, FromCameraCommand = {ToCameraWorkCommand}")

                                # PLC NG
                                try:
                                    client.write_by_name("GVL.CameraWorkStatus", 2) 
                                    client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                                except:
                                    success = False
                                    return success
                        else:

                            # type failed
                            self.logger.error(f"type failed, write CameraWorkStatus = 2, FromCameraCommand = {ToCameraWorkCommand}")

                            # PLC NG
                            try:
                                client.write_by_name("GVL.CameraWorkStatus", 2)  
                                client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                            except:
                                success = False
                                return success
                    else:
                        # request failed
                        self.logger.error(f"request failed, write CameraWorkStatus = 2, FromCameraCommand = {ToCameraWorkCommand}")

                        # PLC NG
                        try:
                            client.write_by_name("GVL.CameraWorkStatus", 2)  
                            client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                        except:
                            success = False
                            return success
                    ################ start video task ################

            elif CameraWorkStatus == 1:

                ################ wait video task complete ################
                if not self.camera.video_task.is_running:
                    # task completed!
                    self.logger.debug("video task completed, write CameraWorkStatus = 0")
                    
                    try:
                        client.write_by_name("GVL.CameraWorkStatus", 0)  
                    except:
                        success = False
                        return success
                else:
                    self.logger.debug("waiting for video task completed!")
                ################ wait video task complete ################
        
        elif ToCameraWorkCommand == 12: # initial task
            
            if CameraWorkStatus == 0: 

                if FromCameraWorkCommand == 0:

                    ################ start initial task ################
                    self.logger.debug("start initial task, write FromCameraWorkCommand = 12, and write CameraWorkStatus = 1")

                    self.camera.initial_task.is_running = True # change status

                    url = "http://127.0.0.1:8080/initial/start_initial_task"
                    ret, response = self.do_tasks(url)
                    self.logger.debug(f"request result----ret:{ret}, response:{response}")
                    if ret and isinstance(response, Iterable):
                        if "status" in response and "message" in response:
                            if response.get("status"):
                                
                                # PLC OK
                                try:
                                    client.write_by_name("GVL.CameraWorkStatus", 1)  
                                    client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                                except:
                                    success = False
                                    return success
                                else:
                                    self.logger.debug("start initial task successfully!")
                            else:

                                # status failed
                                self.logger.error(f"status failed, write CameraWorkStatus = 2, FromCameraCommand = {ToCameraWorkCommand}")

                                # PLC NG
                                try:
                                    client.write_by_name("GVL.CameraWorkStatus", 2)
                                    client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                                except:
                                    success = False
                                    return success
                        else:

                            # type failed
                            self.logger.error(f"type failed, write CameraWorkStatus = 2, FromCameraCommand = {ToCameraWorkCommand}")

                            # PLC NG
                            try:
                                client.write_by_name("GVL.CameraWorkStatus", 2)  # CameraWorkStatus????1,?????
                                client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                            except:
                                success = False
                                return success
                    else:
                        # request failed
                        self.logger.error(f"request failed, write CameraWorkStatus = 2, FromCameraCommand = {ToCameraWorkCommand}")

                        # PLC NG
                        try:
                            client.write_by_name("GVL.CameraWorkStatus", 2)  
                            client.write_by_name("GVL.FromCameraWorkCommand", ToCameraWorkCommand)
                        except:
                            success = False
                            return success
                    ################ start initial task ################

            elif CameraWorkStatus == 1:

                ################ wait initial task complete ################
                if not self.camera.initial_task.is_running:
                    # task completed!
                    self.logger.debug("initial task completed, write CameraWorkStatus = 0")
                    try:
                        client.write_by_name("GVL.CameraWorkStatus", 0) 
                    except:
                        success = False
                        return success
                else:
                    self.logger.debug("waiting for initial task completed!")
                ################ wait initial task complete ################
        
        elif ToCameraWorkCommand == 0: # completed task / no task

            if CameraWorkStatus == 0:

                if FromCameraWorkCommand != 0: # 7, 8, 9, 10, 11, 12

                    ################ complete the task ################
                    self.logger.debug("complete the task, write FromCameraWorkCommand = 0")
                    try:
                        client.write_by_name("GVL.FromCameraWorkCommand", 0)
                    except:
                        success = False
                        return success
                    else:
                        # self.initialize_state = False # initialize
                        pass
                    ################ complete the task ################

            else: # CameraWorkStatus = 1, 2

                ################ abnormal case, reset status ################
                self.logger.error("abnormal case, reset status")
                try:
                    client.write_by_name("GVL.CameraWorkStatus", 0)
                    client.write_by_name("GVL.FromCameraWorkCommand", 0)
                except:
                    success = False
                    return success
                self.initialize_state = False # initialize
                ################ abnormal case, rest status ################

        self.LastRealSenseObstacleSignal = self.RealSenseObstacleSignal # record the last signal 

        return success

    def do_tasks(self, task_url):
        """執行任務"""
        success, data = False, None
        data = {"requestor":"AGVC"}
        try:
            # response = requests.post(url=task_url).json()
            response = requests.post(url=task_url, json=data).json()
        except:
            self.logger.error("request failed!")
        else:
            success = True
            data = response
        return success, data  # 返回執行結果，與取得資料

    def add_camera(self, camera):
        """add camera"""
        self.camera = camera
    
    def move_amr(self, cmd):
        """move amr"""
        success = False
        self.logger.debug(f"move amr, command:{cmd}")
        try:
            if cmd == "stop":
                self.logger.debug("stop amr!")
                try:
                    self.client.write_by_name("GVL.bWeb_ManualSemiControlEnable", 0) # flag
                    self.client.write_by_name("GVL.bWeb_ManualSemiControlForwardButton", 0) # forward stop
                    self.client.write_by_name("GVL.bWeb_ManualSemiControlLeftButton", 0) # left turn stop
                    self.client.write_by_name("GVL.bWeb_ManualSemiControlRightButton", 0) # right turn stop
                except Exception as e:
                    self.logger.error("manual stop control failed!")
                    #send stop until successful
                    if not self.stop_thread.is_alive():
                        self.stop_thread = Thread(target=self.stop_amr_until_successful)
                        self.stop_thread.start()
            elif cmd == "forward":
                self.logger.debug("move forward!")
                try:
                    self.client.write_by_name("GVL.bWeb_ManualSemiControlEnable", 1) # flag
                    self.client.write_by_name("GVL.bWeb_ManualSemiControlForwardButton", 1) # forward
                except Exception as e:
                    self.logger.error("manual forward control failed!")
            elif cmd == "backward":
                self.logger.debug("move backward!")
                try:
                    self.client.write_by_name("GVL.bWeb_ManualSemiControlEnable", 1) # flag
                    self.client.write_by_name("GVL.bWeb_ManualSemiControlBackwardButton", 1) # backward
                except Exception as e:
                    self.logger.error("manual forward control failed!")
            elif cmd == "left-turn":
                self.logger.debug("left-turn!")
                try:
                    self.client.write_by_name("GVL.bWeb_ManualSemiControlEnable", 1) # flag
                    self.client.write_by_name("GVL.bWeb_ManualSemiControlLeftButton", 1) # left turn
                except Exception as e:
                    self.logger.error("manual left-turn control failed!")
            elif cmd == "right-turn":
                self.logger.debug("right-turn!")
                try:
                    self.client.write_by_name("GVL.bWeb_ManualSemiControlEnable", 1) # flag
                    self.client.write_by_name("GVL.bWeb_ManualSemiControlRightButton", 1) # right turn
                except Exception as e:
                    self.logger.error("manual right-turn control failed!")
            else:
                self.logger.error(f"{cmd} is not a valid command!")
        except Exception as e:
            self.logger.error(f"write command failed, error:{e}")
        else:
            success = True
        return success
    
    def stop_amr_until_successful(self):
        """stop amr until successful"""
        while True:
            try:
                self.client.write_by_name("GVL.bWeb_ManualSemiControlEnable", 0) # flag
                self.client.write_by_name("GVL.bWeb_ManualSemiControlForwardButton", 0) # forward stop
                self.client.write_by_name("GVL.bWeb_ManualSemiControlLeftButton", 0) # left turn stop
                self.client.write_by_name("GVL.bWeb_ManualSemiControlRightButton", 0) # right turn stop
            except Exception as e:
                self.logger.error("manual stop control failed!")
            else:
                break
    
    def web_control_heartbeat(self):
        """web control heartbeat"""
        while True:
            try:
                if self.client is not None:
                    self.client.write_by_name("GVL.bWeb_ManualSemiControlHartBit", self.web_control_counter)
            except Exception as e:
                self.logger.error(f"write web hearteat failed, error:{e}")
            finally:
                if self.web_control_counter == 0:
                    self.web_control_counter = 1
                else:
                    self.web_control_counter = 0
            time.sleep(0.5)
    
    def download_obstacle_imgs(self):
        """download obstacle img from apache"""
        
        # create folder
        task_id = datetime.now().strftime("%Y%m%d%H%M%S")
        folder_path = "obstacle_imgs" + os.sep + task_id
        try:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        except Exception as e:
            self.logger.error(f"create obstacle task folder failed, error:{e}")
            return
        
        # iumobo apache server
        front_camera_url = f"http://192.168.0.100/front_camera/current_img.jpg?time={time.time()}"
        back_camera_url = f"http://192.168.0.100/back_camera/current_img.jpg?time={time.time()}"
        left_camera_url = f"http://192.168.0.100/left_camera/current_img.jpg?time={time.time()}"
        right_camera_url = f"http://192.168.0.100/right_camera/current_img.jpg?time={time.time()}"
        urls = [front_camera_url, back_camera_url, left_camera_url, right_camera_url]
        files = ["front_camera.jpg", "back_camera.jpg", "left_camera.jpg", "right_camera.jpg"]
        for item in list(zip(urls, files)):
            url = item[0]
            try:
                response = requests.get(url, stream=True)
            except Exception as e:
                self.logger.error(f"request failed, error:{e}")
            else:
                if response.status_code == 200:
                    local_filname = folder_path + os.sep + item[1]
                    with open(local_filname, "wb") as f:
                        for chunck in response.iter_content(chunk_size=8192):
                            f.write(chunck)
                    self.logger.debug(f"File downloaded succesfully as {local_filname}, url:{url}")
                else:
                    self.logger.error(f"Failed to download file, url:{url}. HTTP Status Code:{response.status_code}")
    
    def ftp_upload_obstacle_imgs(self, task_id):
        """upload obstacle img to ftp server"""
        success = False
        with MyFTP() as ftp:
            # login 
            try:
                ftp.connect(self.camera.ftp.ftp_ip, port=self.camera.ftp.ftp_port)
                ftp.login(self.camera.ftp.ftp_account, self.camera.ftp.ftp_password)
            except:
                self.logger.error("login ftp failed!")
            else:
                self.logger.debug("login successfully!")

                # obstacle_imgs
                ret_base, base_folder = ftp.get_files_list()
                self.logger.debug(f"ret_base:{ret_base}, base_folder:{base_folder}")
                if ret_base:
                    if '/obstacle_imgs' not in base_folder:
                        # create obstacle_imgs folder
                        self.logger.debug("make obstacle_imgs folder.")
                        ftp.make_dir("obstacle_imgs")
                    
                    # enter obstacle_imgs folder
                    if ftp.change_dir("obstacle_imgs"):
                        self.logger.debug(f"ftp cwd:{ftp.get_cwd()}")

                        # task ID
                        ret_task, task_folder = ftp.get_files_list()
                        if ret_task:
                            
                            # create task folder
                            if task_id not in task_folder:
                                self.logger.debug(f"make task_folder:{task_id}.")
                                ftp.make_dir(task_id)
                            
                            # enter task ID folder
                            if ftp.change_dir(task_id):
                                # 已進入工作目錄
                                ret, remote_files = ftp.get_files_list()
                                # 獲取需上傳照片
                                tasks = []
                                local_files = os.listdir("obstacle_imgs" + os.sep + task_id)
                                for local_file in local_files:
                                    if not local_file in remote_files:
                                        tasks.append("obstacle_imgs" + os.sep + task_id + os.sep + local_file)
                                self.logger.debug(f"tasks count:{len(tasks)}.")
                                if len(tasks) > 0:
                                    success_cnt = 0
                                    for task in tasks:
                                        if ftp.upload_file(task, os.path.basename(task)):
                                            success_cnt += 1
                                            self.logger.debug(f"upload ok, success_cnt:{success_cnt}.")
                                        else:
                                            self.logger.error(f"upload failed, success_cnt:{success_cnt}, stopped!")
                                            break  
                                    if success_cnt == len(tasks):
                                        success = True
                                else:
                                    self.logger.debug(f"no images needed to be upload!")
                                    success = True
                            else:
                                self.logger.error("change to task id folder failed!")
                        else:
                            self.logger.error("retrieve task id folder failed!")
                    else:
                        self.logger.error("change to obstancle_imgs folder failed!")
                else:
                    self.logger.error("retrieve obstancle_imgs folder failed!")
        return success
    
    def ftp_remove_obstacle_imgs(self, task_id):
        """FTP上傳完成刪除照片"""
        self.logger.debug(f"ftp remove imgs, task_id:{task_id}...")
        status, message = False, ""
        if task_id:
            dir = "./obstacle_imgs" + os.sep + task_id
            self.logger.debug(f"remove dir:{dir}")
            if os.path.exists(dir):
                # 刪檔
                files = os.listdir(dir)
                self.logger.debug(f"try to remove files:{files}")
                success1 = True
                for file in files:
                    self.logger.debug(f"try to remove file:{file}")
                    try:
                        os.remove(dir + os.sep + file)
                    except:
                        self.logger.error(f"remove file:{file} failed!")
                        success1 = False
                        message = "remove img file failed!"

                # 刪task資料夾
                success2 = True
                self.logger.debug(f"try to remove task dir:{dir}")
                try:
                    os.rmdir(dir)
                except Exception as e:
                    self.logger.debug(f"remove directory failed!")
                    success2 = False
                    message = f"remove task directory failed!"

                if success1 and success2:
                    self.logger.debug(f"remove successfully!")
                    status = True
            else:
                # 無此檔案
                message = "directory not found!"
                self.logger.debug(f"directory not found!")

        self.logger.debug(f"ftp remove obstacle imgs complete, satus:{status}, message:{message}")
        return status, message

    def download_and_upload_obstacle_imgs(self):
        """download and upload obstacle imgs"""
        self.logger.debug(f"download and upload obstacle imgs...")
        
        # download obstacle imgs
        self.download_obstacle_imgs() # download obstacle imgs
        base_dir = "./obstacle_imgs" # get task
        self.logger.debug(f"base_dir:{base_dir}")
        try:
            tasks = os.listdir(base_dir)
        except Exception as e:
            self.logger.error(f"listdir error:{e}")
        else:
            for task_id in tasks: 
                # upload obstacle imgs
                if self.ftp_upload_obstacle_imgs(task_id):
                    self.logger.debug(f"upload obstacle imgs successfully!")
                    # remove obstacle 
                    self.ftp_remove_obstacle_imgs(task_id)
                else:
                    self.logger.error(f"upload obstacle imgs failed!")
                    break


def stitch_target_image(dir) -> bool:
    """stitch target images"""

    success = False
    start = time.time()
    print(f"dir:{dir}")

    try:
        files = os.listdir(dir)
    except:
        return success
    else:
        for file in files:
            file_name, file_extension = os.path.splitext(file)
            if file_extension != ".jpg":
                files.remove(file)
        print(files)
        pano = stitch(dir, files)

        RESULT = dir + os.sep + 'target.jpg'
        if pano is not None:
            success = True
            pano = crop(pano)
            pano = add_black_margin(pano)
            try:
                cv2.imwrite(RESULT, pano)
            except:
                print("save target image failed!")
        else:
            print("stitch error")        
        
        end = time.time()
        print("cost "+str(end-start))
    print(success)
    return success


def convert_angle(angle:int):
    """convert angle to ~ 180 degrees"""
    while angle < 0:
        angle += 360
    angle %= 360 
    return angle


def calculate_ptz_value(ptz):
    """
    角度轉轉換PTZ值：
        pan：-170 ~ 170度，對應-1.0 ~ 1.0
        tilt：-30 ~ 90度，對應-0.3 ~ 0.9
        zoom：0.0 ~ 1.0，對應0 ~ 12X 變焦
    """
    if ptz[0] < -170 or ptz[0] > 170:
        raise ValueError("pan角度範圍應在-170 ~ 170之間!")
    if ptz[1] < -30 or ptz[1] > 90:
        raise ValueError("tilt角度範圍應在-30 ~ 90之間!")
    if ptz[2] < 0 or ptz[2] > 1:
        raise ValueError("zoom倍率範圍應在0 ~ 1之間!")

    pan_value = ptz[0] * (1/170)
    tilt_value = ptz[1] / 100
    zoom_value = ptz[2]
    return [pan_value, tilt_value, zoom_value]


def calculate_ptz_angle(ptz):
    """
    PTZ值轉換角度：
        pan：-170 ~ 170度，對應-1.0 ~ 1.0
        tilt：-30 ~ 90度，對應-0.3 ~ 0.9
        zoom：0.0 ~ 1.0，對應0 ~ 12X 變焦
    """
    if ptz[0] < -1.0 or ptz[0] > 1.0:
        raise ValueError("pan角度範圍應在-1.0 ~ 1.0之間!")
    if ptz[1] < -0.3 or ptz[1] > 0.9:
        raise ValueError("tilt角度範圍應在0.3 ~ 0.9之間!")
    if ptz[2] < 0 or ptz[2] > 1:
        raise ValueError("zoom倍率範圍應在0 ~ 1之間!")

    pan_angle = (ptz[0] * 170) // 1
    tilt_angle = (ptz[1] * 100)  // 1
    zoom_value = round(ptz[2], 1) # maybe wroung
    return [pan_angle, tilt_angle, zoom_value]


def move_to_abs(camera, ptz_angle=[0.0, 0.0, 0.0], wait_seconds=20):
    """PTZ移動至指定角度位置"""
    success = False
    start = time.time()
    end = start + wait_seconds
    try:
        ptz_value = calculate_ptz_value(ptz_angle)  # 角度換算成控制值
    except:
        print("calculate ptz value failed!")
        return
    ret = camera.onvif.abs_move(ptz_value[0], ptz_value[1], ptz_value[2])
    while ret:

        # timeout
        now = time.time()
        if now >= end:
            # print("Timeout!")
            # 更新PTZ
            camera.crr_pan = None
            camera.crr_tilt = None
            camera.crr_zoom = None
            break

        # 獲取PTZ值
        ptz_status = camera.onvif.get_ptz_status()
        pan, tilt, zoom = ptz_status.get("pan"), \
            ptz_status.get("tilt"), \
            ptz_status.get("zoom")

        if isinstance(pan, float) and isinstance(tilt, float) and isinstance(zoom, float):
            # 檢查是否到位
            pan_diff, tilt_diff, zoom_diff = abs(ptz_value[0]-pan), \
                abs(ptz_value[1]-tilt), \
                abs(ptz_value[2]-zoom)
            if pan_diff <= 0.005 and tilt_diff <= 0.005 and zoom_diff <= 0.005:
                # print("Move completed!")
                success = True

                # 更新PTZ
                camera.crr_pan = ptz_angle[0]
                camera.crr_tilt = ptz_angle[1]
                camera.crr_zoom = ptz_angle[2]
                break
        else:
            print("get ptz value failed!")
            break

    time.sleep(1)  # 等待1秒

    return success


def create_task_folder(camera):
    """建立task資料夾"""
    base_folder = "save_imgs"
    pos_folder = base_folder + os.sep + camera.pos_folder
    task_folder = pos_folder + os.sep + camera.task_folder
    try:
        # 建立 POS 位置資料夾
        if not os.path.exists(pos_folder):
            os.makedirs(pos_folder)
        # 建立 TASK 時間資料夾
        if not os.path.exists(task_folder):
            os.makedirs(task_folder)
    except Exception as e:
        print(e.args)
    return task_folder


def run_initial_task(camera):
    """Initaial Task """
    camera.initial_task.is_running = True
    
    while camera.initial_task.qsize() > 0:

        ptz_angle = camera.initial_task.get()

        ret = move_to_abs(camera, ptz_angle)  # PTZ控制
        if not ret:
            print(f"ptz:{ptz_angle} retrived image failed!")

        time.sleep(0.5)
    
    camera.initial_task.is_running = False


def run_panorama_task(camera):
    """Panorama 全景環景拍攝"""
    task_folder = create_task_folder(camera)  # 建立task資料夾
    filename = task_folder + os.sep + "info.txt"
    start_time = time.time()
    write_file(filename, f"amr_pos_theta:{camera.amr.amr_pos_theta}\n")
    write_file(filename, f"camera_offset:{camera.camera_offset}\n")
    write_file(filename, f"task_type:panorama\n")
    write_file(filename, f"task_cnt:{camera.panorama_task.qsize()}\n")
    write_file(filename, f"requestor:{camera.task_requestor}\n")
    write_file(filename, f"amr_pos_x:{camera.amr.amr_pos_x}\n")
    write_file(filename, f"amr_pos_y:{camera.amr.amr_pos_y}\n")
    write_file(filename, f"amr_pos_z:{camera.amr.amr_pos_z}\n")
    write_file(filename, f"amr_tag_id:{camera.amr.amr_tag_id}\n")
    write_file(filename, f"ftp_url:{task_folder}\n")
    write_file(filename, f'task_time:{camera.panorama_task.start_time.strftime("%Y-%m-%d %H:%M:%S")}\n')

    camera.panorama_task.is_running = True
    camera.panorama_task.stop_flag = False
    
    # run initial task
    # run_initial_task(camera)

    while camera.panorama_task.qsize() > 0:

        if camera.panorama_task.stop_flag:
            # 中斷環景拍攝任務
            print("panorama task stopped!")
            break

        ptz_angle = camera.panorama_task.get()

        ret = move_to_abs(camera, ptz_angle)  # PTZ控制
        if not ret:
            print(f"ptz:{ptz_angle} retrived image failed!")
        else:
            save_img(camera, ptz_angle)  # 存照片

        print(f"panorama task left cnt:{camera.panorama_task.qsize()}")

        time.sleep(0.5)

    write_file(
        filename, f"task_left:{camera.panorama_task.qsize()}\n")
    
    # clear initial task
    if not camera.initial_task.empty():
        camera.initial_task.queue.clear()
    # 清空任務
    if not camera.panorama_task.empty():
        camera.panorama_task.queue.clear()
    write_file(filename, f"time_cost:{int(time.time()-start_time)}\n")

    # camera.panorama_task.is_running = False

    # ftp上傳圖像
    print("uploading imgs...")
    print(f"pos_id:{camera.pos_folder}, task_folder:{camera.task_folder}")
    ret = ftp_upload_imgs(camera, camera.pos_folder, camera.task_folder)
    if ret:
        # 上傳成功刪圖像
        ftp_remove_imgs(camera, camera.pos_folder, camera.task_folder)

        # write to mysql
        query="""
insert into `task_history` (`task_type`, `amr_pos_X`, `amr_pos_y`, `amr_pos_z`, 
`amr_pos_theta`, `amr_tag_id`, `ftp_url`, `task_time`, `stitch_state`, `requestor`) values 
(%s, %s, %s, %s, %s, %s, %s, %s, %s )
"""
        print(query)
        data = ("panorama", camera.pos_folder_x, camera.pos_folder_y, camera.pos_folder_z, 
                camera.pos_folder_theta, camera.pos_folder_tag_id, "save_imgs"+os.sep+camera.pos_folder+os.sep+camera.task_folder, 
                camera.panorama_task.start_time.strftime("%Y-%m-%d %H:%M:%S"), "none", camera.task_requestor)
        camera.mysql_conn.UpdateRowsByTuple(query, data)
    else:
        print("uploading imgs failed!")
    
    if ret:
        # 補上傳其它任務圖像
        ftp_upload_previous_tasks(camera)

    print("panorama task finished!")
    camera.panorama_task.is_running = False


def run_target_task(camera):
    """Target環景拍攝"""
    task_folder = create_task_folder(camera)  # 建立task資料夾
    filename = task_folder + os.sep + "info.txt"
    start_time = time.time()
    write_file(filename, f"amr_pos_theta:{camera.amr.amr_pos_theta}\n")
    write_file(filename, f"camera_offset:{camera.camera_offset}\n")
    write_file(filename, f"task_type:target\n")
    write_file(filename, f"task_cnt:{camera.target_task.qsize()}\n")
    write_file(filename, f"requestor:{camera.task_requestor}\n")
    write_file(filename, f"amr_pos_x:{camera.amr.amr_pos_x}\n")
    write_file(filename, f"amr_pos_y:{camera.amr.amr_pos_y}\n")
    write_file(filename, f"amr_pos_z:{camera.amr.amr_pos_z}\n")
    write_file(filename, f"amr_tag_id:{camera.amr.amr_tag_id}\n")
    write_file(filename, f"ftp_url:{task_folder}\n")
    write_file(filename, f'task_time:{camera.target_task.start_time.strftime("%Y-%m-%d %H:%M:%S")}\n')

    camera.target_task.is_running = True
    camera.target_task.stop_flag = False

    # run initial task
    # run_initial_task(camera)

    while camera.target_task.qsize() > 0:

        if camera.target_task.stop_flag:
            # 中斷Target環景拍攝任務
            print("target task stopped!")
            break

        ptz_angle = camera.target_task.get()

        ret = move_to_abs(camera, ptz_angle)  # PTZ控制
        if not ret:
            print(f"ptz:{ptz_angle} retrived image failed!")
        else:
            save_img(camera, ptz_angle)  # 存照片

        print(f"target task left cnt:{camera.target_task.qsize()}")

        time.sleep(0.5)

    write_file(
        filename, f"task_left:{camera.target_task.qsize()}\n")
    
    # clear initial task
    if not camera.initial_task.empty():
        camera.initial_task.queue.clear()
    # 清空任務
    if not camera.target_task.empty():
        camera.target_task.queue.clear()

    # stitch images, maybe failed and raised exception
    # try:
    #     if not stitch_target_image("save_imgs"+ os.sep + camera.pos_folder + os.sep + camera.task_folder):
    #         print("stitch target images failed!")
    #         write_file(filename, f"stitch_state:ng\n")
    #     else:
    #         write_file(filename, f"stitch_state:ok\n")
    # except:
    #     write_file(filename, f"stitch_state:ng\n")
    write_file(filename, f"stitch_state:none\n")
    
    # read info.txt and get stitch state
    lines = read_lines("save_imgs"+ os.sep + camera.pos_folder + os.sep + camera.task_folder + os.sep + "info.txt") 
    stitch_state = "none"
    for line in lines:
        if "stitch_state:" in line:
            stitch_state = line.replace("stitch_state:", "")
    write_file(filename, f"time_cost:{int(time.time()-start_time)}\n")

    # ftp上傳圖像
    print("uploading imgs...")
    print(f"pos_id:{camera.pos_folder}, task_folder:{camera.task_folder}")
    ret = ftp_upload_imgs(camera, camera.pos_folder, camera.task_folder)
    if ret:
        # write to mysql
        query="""
insert into `task_history` (`task_type`, `amr_pos_X`, `amr_pos_y`, `amr_pos_z`, 
`amr_pos_theta`, `amr_tag_id`, `ftp_url`, `task_time`, `stitch_state`, `requestor`) values 
(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s )
"""
        # print(query)
        data = ("target", camera.pos_folder_x, camera.pos_folder_y, camera.pos_folder_z, 
                camera.pos_folder_theta, camera.pos_folder_tag_id, "save_imgs"+os.sep+camera.pos_folder+os.sep+camera.task_folder, 
                camera.target_task.start_time.strftime("%Y-%m-%d %H:%M:%S"), stitch_state, camera.task_requestor)
        if camera.mysql_conn.UpdateRowsByTuple(query, data):
            # 上傳成功刪圖像
            ftp_remove_imgs(camera, camera.pos_folder, camera.task_folder)
    else:
        print("uploading imgs failed!")

    if ret:
        # 補上傳其它任務圖像
        ftp_upload_previous_tasks(camera)    

    print("target task finished!")
    camera.target_task.is_running = False


def run_designated_task(camera):
    """designated指定點拍攝"""
    task_folder = create_task_folder(camera)  # 建立task資料夾
    filename = task_folder + os.sep + "info.txt"
    start_time = time.time()
    write_file(filename, f"amr_pos_theta:{camera.amr.amr_pos_theta}\n")
    write_file(filename, f"camera_offset:{camera.camera_offset}\n")
    write_file(filename, f"task_type:designated\n")
    write_file(filename, f"task_cnt:{camera.designated_task.qsize()}\n")
    write_file(filename, f"requestor:{camera.task_requestor}\n")
    write_file(filename, f"amr_pos_x:{camera.amr.amr_pos_x}\n")
    write_file(filename, f"amr_pos_y:{camera.amr.amr_pos_y}\n")
    write_file(filename, f"amr_pos_z:{camera.amr.amr_pos_z}\n")
    write_file(filename, f"amr_tag_id:{camera.amr.amr_tag_id}\n")
    write_file(filename, f"ftp_url:{task_folder}\n")
    write_file(filename, f'task_time:{camera.designated_task.start_time.strftime("%Y-%m-%d %H:%M:%S")}\n')

    camera.designated_task.is_running = True
    camera.designated_task.stop_flag = False
    
    # run initial task
    # run_initial_task(camera)

    while camera.designated_task.qsize() > 0:

        if camera.designated_task.stop_flag:
            # 中斷designated 拍攝任務
            print("designated task stopped!")
            break

        ptz_angle = camera.designated_task.get()

        ret = move_to_abs(camera, ptz_angle)  # PTZ控制
        if not ret:
            print(f"ptz:{ptz_angle} retrived image failed!")
        else:
            save_img(camera, ptz_angle)  # 存照片

        print(f"designated task left cnt:{camera.designated_task.qsize()}")

        time.sleep(0.5)

    write_file(
        filename, f"task_left:{camera.designated_task.qsize()}\n")
    
    # clear initial task
    if not camera.initial_task.empty():
        camera.initial_task.queue.clear()
    # 清空任務
    if not camera.designated_task.empty():
        camera.designated_task.queue.clear()
    
    write_file(filename, f"stitch_state:none\n")    

    write_file(filename, f"time_cost:{int(time.time()-start_time)}\n")

    # ftp上傳圖像
    print("uploading imgs...")
    print(f"pos_id:{camera.pos_folder}, task_folder:{camera.task_folder}")
    ret = ftp_upload_imgs(camera, camera.pos_folder, camera.task_folder)
    if ret:
        # write to mysql
        query="""
insert into `task_history` (`task_type`, `amr_pos_X`, `amr_pos_y`, `amr_pos_z`, 
`amr_pos_theta`, `amr_tag_id`, `ftp_url`, `task_time`, `stitch_state`, `requestor`) values 
(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s )
"""
        print(query)
        data = ("designated", camera.pos_folder_x, camera.pos_folder_y, camera.pos_folder_z, 
                camera.pos_folder_theta, camera.pos_folder_tag_id, "save_imgs"+os.sep+camera.pos_folder+os.sep+camera.task_folder, 
                camera.designated_task.start_time.strftime("%Y-%m-%d %H:%M:%S"), "none", camera.task_requestor)
        if camera.mysql_conn.UpdateRowsByTuple(query, data):
            # 上傳成功刪圖像
            ftp_remove_imgs(camera, camera.pos_folder, camera.task_folder)
    else:
        print("uploading imgs failed!")
    
    if ret:
        # 補上傳其它任務圖像
        ftp_upload_previous_tasks(camera)   

    print("designated task finished!")
    camera.designated_task.is_running = False


def run_ir_task(camera):
    """熱顯像儀拍照"""
    # print("run ir task")
    camera.main_logger.debug("run ir task!")

    task_folder = create_task_folder(camera)  # 建立task資料夾
    filename = task_folder + os.sep + "info.txt"
    start_time = time.time()
    write_file(filename, f"amr_pos_theta:{camera.amr.amr_pos_theta}\n")
    write_file(filename, f"camera_offset:{camera.camera_offset}\n")
    write_file(filename, f"task_type:ir\n")
    write_file(filename, f"task_cnt:{2}\n")
    write_file(filename, f"requestor:{camera.task_requestor}\n")
    write_file(filename, f"amr_pos_x:{camera.amr.amr_pos_x}\n")
    write_file(filename, f"amr_pos_y:{camera.amr.amr_pos_y}\n")
    write_file(filename, f"amr_pos_z:{camera.amr.amr_pos_z}\n")
    write_file(filename, f"amr_tag_id:{camera.amr.amr_tag_id}\n")
    write_file(filename, f"ftp_url:{task_folder}\n")
    write_file(filename, f'task_time:{camera.ir_task.start_time.strftime("%Y-%m-%d %H:%M:%S")}\n')

    camera.ir_task.is_running = True

    try:
        cv2.imwrite(task_folder+os.sep+"ir.jpg", camera.ir_cam.get_img())
        cv2.imwrite(task_folder+os.sep+"ir-colormap.jpg", camera.ir_cam.get_colormap_img(mark_max_temp=True))
    except Exception as e:
        print(e.args)
        camera.main_logger.error(f"error:{e.args}")

    # camera.ir_task.is_running = False
    write_file(
        filename, f"task_left:{0}\n")

    write_file(filename, f"time_cost:{int(time.time()-start_time)}\n")

    write_file(filename, f"stitch_state:none\n")

    # ftp上傳圖像
    print("uploading imgs...")
    print(f"pos_id:{camera.pos_folder}, task_folder:{camera.task_folder}")
    camera.main_logger.debug(f"uploading imgs..., pos_id:{camera.pos_folder}, task_folder:{camera.task_folder}")
    ret = ftp_upload_imgs(camera, camera.pos_folder, camera.task_folder)
    if ret:
        camera.main_logger.debug(f"uploading imgs successfully!")
        # write to mysql
        query="""
insert into `task_history` (`task_type`, `amr_pos_X`, `amr_pos_y`, `amr_pos_z`, 
`amr_pos_theta`, `amr_tag_id`, `ftp_url`, `task_time`, `stitch_state`, `requestor`) values 
(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s )
"""
        # print(query)
        camera.main_logger.debug(f"query:{query}")
        data = ("ir", camera.pos_folder_x, camera.pos_folder_y, camera.pos_folder_z, 
                camera.pos_folder_theta, camera.pos_folder_tag_id, "save_imgs"+os.sep+camera.pos_folder+os.sep+camera.task_folder, 
                camera.ir_task.start_time.strftime("%Y-%m-%d %H:%M:%S"), "none", camera.task_requestor)
        if camera.mysql_conn.UpdateRowsByTuple(query, data):
            # 上傳成功刪圖像
            camera.main_logger.debug(f"mysql insert successfully!")
            camera.main_logger.debug("ftp remove imgs")
            ftp_remove_imgs(camera, camera.pos_folder, camera.task_folder)
    else:
        print("uploading imgs failed!")
        camera.main_logger.debug(f"uploading imgs failed!")
    
    # 補上傳其它任務圖像
    if ret:
        ftp_upload_previous_tasks(camera)

    print("ir task finished!")
    camera.main_logger.debug(f"ir task finished!")
    camera.ir_task.is_running = False


def run_video_task(camera):
    """make video"""
    camera.main_logger.debug("run video task!")
    task_folder = create_task_folder(camera)  # 建立task資料夾
    filename = task_folder + os.sep + "info.txt"
    start_time = time.time()
    write_file(filename, f"amr_pos_theta:{camera.amr.amr_pos_theta}\n")
    write_file(filename, f"camera_offset:{camera.camera_offset}\n")
    write_file(filename, f"task_type:video\n")
    write_file(filename, f"task_cnt:{camera.video_task.qsize()}\n")
    write_file(filename, f"requestor:{camera.task_requestor}\n")
    write_file(filename, f"amr_pos_x:{camera.amr.amr_pos_x}\n")
    write_file(filename, f"amr_pos_y:{camera.amr.amr_pos_y}\n")
    write_file(filename, f"amr_pos_z:{camera.amr.amr_pos_z}\n")
    write_file(filename, f"amr_tag_id:{camera.amr.amr_tag_id}\n")
    write_file(filename, f"ftp_url:{task_folder}\n")
    write_file(filename, f'task_time:{camera.video_task.start_time.strftime("%Y-%m-%d %H:%M:%S")}\n')

    camera.video_task.is_running = True
    camera.video_task.stop_flag = False
    
    # run initial task
    # run_initial_task(camera)
    video_index = 0
    while camera.video_task.qsize() > 0:

        if camera.video_task.stop_flag:
            # 中斷 video 拍攝任務
            camera.main_logger.error("video task stopped!")
            break

        ptz_angle = camera.video_task.get()
        pan, tilt, zoom, video_time = ptz_angle[0], ptz_angle[1], ptz_angle[2], ptz_angle[3]
        camera.main_logger.debug(f"pan:{pan}, tilt:{tilt}, zoom:{zoom}, video_time:{video_time}")
        ptz_angle = [pan, tilt, zoom]
        camera.main_logger.debug(f"ptz_angle:{ptz_angle}, video_time:{video_time}")
        ret = move_to_abs(camera, ptz_angle)  # PTZ控制
        if not ret:
            # print(f"ptz:{ptz_angle} retrived image failed!")
            camera.main_logger.error(f"ptz:{ptz_angle} retrieved image failed!")
        else:
            # make video
            camera.main_logger.debug(f"make video for {video_time}s")
            camera.make_video(video_time, task_folder + os.sep + f"output_{video_index}.mp4")
            try:
                camera.video_timer.join() 
            except Exception as e:
                pass
                camera.main_logger.error(f"Error:{e.args}")
        video_index += 1

        # print(f"video task left cnt:{camera.video_task.qsize()}")
        camera.main_logger.error(f"video task left cnt:{camera.video_task.qsize()}")

        time.sleep(0.5)

    write_file(
        filename, f"task_left:{camera.video_task.qsize()}\n")
    
    # clear initial task
    if not camera.initial_task.empty():
        camera.initial_task.queue.clear()
    # 清空任務
    if not camera.video_task.empty():
        camera.video_task.queue.clear()
    
    write_file(filename, f"stitch_state:none\n")    
    write_file(filename, f"time_cost:{int(time.time()-start_time)}\n")

    # ftp上傳圖像
    # print("uploading videos...")
    # print(f"pos_id:{camera.pos_folder}, task_folder:{camera.task_folder}")
    camera.main_logger.debug(f"uploading videos..., pos_id:{camera.pos_folder}, task_folder:{camera.task_folder}")
    ret = ftp_upload_imgs(camera, camera.pos_folder, camera.task_folder)
    if ret:
        camera.main_logger.debug(f"uploading videos successfully, and write to mysql.")
        # write to mysql
        query="""
insert into `task_history` (`task_type`, `amr_pos_X`, `amr_pos_y`, `amr_pos_z`, 
`amr_pos_theta`, `amr_tag_id`, `ftp_url`, `task_time`, `stitch_state`, `requestor`) values 
(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s )
"""
        # print(query)
        data = ("video", camera.pos_folder_x, camera.pos_folder_y, camera.pos_folder_z, 
                camera.pos_folder_theta, camera.pos_folder_tag_id, "save_imgs"+os.sep+camera.pos_folder+os.sep+camera.task_folder, 
                camera.video_task.start_time.strftime("%Y-%m-%d %H:%M:%S"), "none", camera.task_requestor)
        if camera.mysql_conn.UpdateRowsByTuple(query, data):
            # 上傳成功刪圖像
            camera.main_logger.debug("insert mysql successfully, ftp remove imgs")
            ftp_remove_imgs(camera, camera.pos_folder, camera.task_folder)
    else:
        camera.main_logger.error("uploading video failed!")
    
    if ret:
        # 補上傳其它任務圖像
        ftp_upload_previous_tasks(camera)   

    # print("video task finished!")
    camera.main_logger.debug("video task finished!")
    camera.video_task.is_running = False


def save_img(camera, ptz_angle):
    """存照片"""
    # 格式：save_imgs/位置(x,y,theta,tag_id)/task年月日時分秒/img_年月日時分秒_pan(0.0)_tilt(0.0)_zoom(0.0).jpg

    img = camera.get_img(True)
    if img is not None:
        task_folder = create_task_folder(camera)
        filename = task_folder + os.sep + \
            f'img_{datetime.now().strftime("%Y%m%d%H%M%S")}_{ptz_angle[0]}_{ptz_angle[1]}_{ptz_angle[2]}.jpg'

        if camera.save_global_coordinate:
            # 存大地坐標, img_時間_x_y_(theta+pan).jpg
            if isinstance(camera.amr.amr_pos_x, int) and isinstance(camera.amr.amr_pos_y, int) and isinstance(camera.amr.amr_pos_theta, int):
                
                # calculate the target 
                theta_converted = convert_angle(camera.amr.amr_pos_theta)
                camera_pos = theta_converted + camera.camera_offset
                camera_pos_converted = convert_angle(camera_pos)
                target_pos = camera_pos_converted - int(ptz_angle[0])
                target_pos_converted = convert_angle(target_pos)

                # filename = task_folder + os.sep + \
                #     f'img_{datetime.now().strftime("%Y%m%d%H%M%S")}_{camera.amr.amr_pos_x}_{camera.amr.amr_pos_y}_{camera.amr.amr_pos_theta+int(ptz_angle[0])}.jpg'
                filename = task_folder + os.sep + \
                    f'img_{datetime.now().strftime("%Y%m%d%H%M%S")}_{camera.amr.amr_pos_x}_{camera.amr.amr_pos_y}_{target_pos_converted}.jpg'
        try:
            # 存檔
            cv2.imwrite(filename, img)
        except Exception as e:
            print(e.args)
    else:
        print("retrieve img failed, no img saved!")


def ftp_upload_imgs(camera, pos_id, task_id):
    """FTP上傳照片"""

    # print(f"starting upload images, pos_id={pos_id}, task={task_id}")
    camera.main_logger.info(f"starting upload images, pos_id={pos_id}, task={task_id}")
    success = False
    with MyFTP() as ftp:
        # login 
        try:
            ftp.connect(camera.ftp.ftp_ip, port=camera.ftp.ftp_port)
            ftp.login(camera.ftp.ftp_account, camera.ftp.ftp_password)
        except:
            camera.main_logger.error("login ftp failed!")
        else:
            camera.main_logger.info("login successfully!")

            # 第一層save_imgs
            ret_base, base_folder = ftp.get_files_list()
            if ret_base:
                if '/save_imgs' not in base_folder:
                    # 建立save_imgs資料夾
                    camera.main_logger.debug("make save_imgs folder.")
                    ftp.make_dir("save_imgs")
                
                # 進入save_img資料夾
                if ftp.change_dir("save_imgs"):
                    camera.main_logger.debug(f"ftp cwd:{ftp.get_cwd()}")

                    # 第二層POS ID
                    ret_pos, pos_folder = ftp.get_files_list()
                    if ret_pos:
                        
                        # 建立POS ID資料夾
                        if pos_id not in pos_folder:
                            camera.main_logger.debug(f"make pos_folder:{pos_id}.")
                            ftp.make_dir(pos_id)
                        
                        # 進入POS ID資料夾
                        if ftp.change_dir(pos_id):

                            # 第三層task id
                            ret_task, task_folder = ftp.get_files_list()
                            if ret_task:
                                if task_id not in task_folder:
                                    # 建立TASK ID資料夾
                                    camera.main_logger.debug(f"make task_folder:{task_id}.")
                                    ftp.make_dir(task_id)
                                
                                # 進入TASK ID資料夾
                                if ftp.change_dir(task_id):

                                    # 已進入工作目錄
                                    ret, remote_files = ftp.get_files_list()
                                    if ret:
                                        # 獲取需上傳照片
                                        tasks = []
                                        local_files = os.listdir("save_imgs" + os.sep + pos_id + os.sep + task_id)
                                        print(f"local_files:{local_files}")
                                        for local_file in local_files:
                                            if not local_file in remote_files:
                                                tasks.append("save_imgs" + os.sep + pos_id + os.sep + task_id + os.sep + local_file)
                                        # print(f"tasks:{tasks}")
                                        camera.main_logger.info(f"tasks count:{len(tasks)}.")
                                        if len(tasks) > 0:
                                            success_cnt = 0
                                            for task in tasks:
                                                if ftp.upload_file(task, os.path.basename(task)):
                                                    success_cnt += 1
                                                    camera.main_logger.debug(f"upload ok, success_cnt:{success_cnt}.")
                                                    # try:
                                                    #     os.remove(task)
                                                    # except:
                                                    #     camera.main_logger.error(f"remove local file failed, file path:{task}")
                                                    #     break
                                                else:
                                                    camera.main_logger.error(f"upload failed, success_cnt:{success_cnt}, stopped!")
                                                    break
                                            if success_cnt == len(tasks):
                                                success = True
                                                camera.main_logger.info("uploaded images successfully!")
                                        else:
                                            camera.main_logger.debug(f"no images needed to be upload!")
                                            success = True
                                    else:
                                        camera.main_logger.error("retrieve remove files failed!")
                                else:
                                    camera.main_logger.error("change to task id folder failed!")
                            else:
                                camera.main_logger.error("retrieve task id folder failed!")
                        else:
                            camera.main_logger.error("change to pos id folder failed!")
                    else:
                        camera.main_logger.error("retrieve pos id folder failed!")
                else:
                    camera.main_logger.error("change to save_imgs folder failed!")                        
            else:
                camera.main_logger.error("retrieve save_imgs folder failed!")
    return success


def ftp_remove_imgs(camera, pos_id, task_id):
    """FTP上傳完成刪除照片"""
    camera.main_logger.debug(f"ftp remove imgs, pos_id:{pos_id}, task_id:{task_id}...")
    status, message = False, ""
    if pos_id and task_id:
        dir = "./save_imgs" + os.sep + pos_id + os.sep + task_id
        camera.main_logger.debug(f"remove dir:{dir}")
        if os.path.exists(dir):
            # 刪檔
            files = os.listdir(dir)
            camera.main_logger.debug(f"try to remove files:{files}")
            success1 = True
            for file in files:
                camera.main_logger.debug(f"try to remove file:{file}")
                try:
                    os.remove(dir + os.sep + file)
                except:
                    # print(f"remove file:{file} failed!")
                    camera.main_logger.error(f"remove file:{file} failed!")
                    success1 = False
                    message = "remove img file failed!"

            # 刪task資料夾
            success2 = True
            camera.main_logger.debug(f"try to remove task dir:{dir}")
            try:
                os.rmdir(dir)
            except:
                # print("remove directory failed!")
                camera.main_logger.error(f"remove directory failed!")
                success2 = False
                message = f"remove task directory failed!"

            # 刪除pos資料夾
            success3 = True
            pos_dir = "./save_imgs" + os.sep + pos_id
            camera.main_logger.debug(f"try to remove pos dir:{pos_dir}")
            files = os.listdir(pos_dir)
            camera.main_logger.debug(f"pos dir files:{files}")
            if not files:
                camera.main_logger.debug(f"try to pos_dir:{pos_dir}")
                try:
                    os.rmdir(pos_dir)
                except:
                    # print("remove pos directory failed")
                    camera.main_logger.error(f"remove pos directory failed")
                    success3 = False
                    message = f"remove pos directory failed"

            if success1 and success2 and success3:
                camera.main_logger.debug(f"remove successfully!")
                status = True
        else:
            # 無此檔案
            message = "directory not found!"
            camera.main_logger.error(f"directory not found!")

    camera.main_logger.debug(f"ftp remove imgs complete, satus:{status}, message:{message}")
    return status, message


def ftp_upload_previous_tasks(camera):
    """ uploading previous failed task imgs """
    camera.main_logger.debug("uploading previous failed task imgs...")
    for pos in os.listdir("save_imgs"):
        for task in os.listdir("save_imgs" + os.sep + pos):
            camera.main_logger.debug(f"pos:{pos}, task:{task}")
            ret1 = ftp_upload_imgs(camera, pos, task)
            if ret1:
                camera.main_logger.debug(f"uploading successfully, pos:{pos}, task:{task}.")
                # read info.txt
                info_txt = "save_imgs"+ os.sep + pos + os.sep + task + os.sep + "info.txt"
                camera.main_logger.debug(f"read info.txt, path{info_txt}.")
                try:
                    lines = read_lines("save_imgs"+ os.sep + pos + os.sep + task + os.sep + "info.txt") 
                except Exception as e:
                    camera.main_logger.error(f"read task folder failed, pos:{pos}, task:{task}, error:{e}.")
                else:
                    camera.main_logger.debug(f"read info.txt OK!")
                    task_type = ""
                    amr_pos_x, amr_pos_y, amr_pos_z, amr_pos_theta, amr_tag_id = None, None, None, None, None
                    ftp_url = ""
                    stitch_state = "none"
                    requestor = "manual"
                    task_time = "0000-00-00 00:00:00"
                    for line in lines:
                        if "task_type" in line:
                            task_type = line.replace("task_type:", "")
                        if "amr_pos_x" in line:
                            amr_pos_x = line.replace("amr_pos_x:", "")
                        if "amr_pos_y" in line:
                            amr_pos_y = line.replace("amr_pos_y:", "")
                        if "amr_pos_z" in line:
                            amr_pos_z = line.replace("amr_pos_z:", "")
                        if "amr_pos_theta" in line:
                            amr_pos_theta = line.replace("amr_pos_theta:", "")
                        if "amr_tag_id" in line:
                            amr_tag_id = line.replace("amr_tag_id:", "")
                        if "ftp_url" in line:
                            ftp_url = line.replace("ftp_url:", "")                     
                        if "stitch_state:" in line:
                            stitch_state = line.replace("stitch_state:", "")
                        if "requestor" in line:
                            requestor = line.replace("requestor:", "")
                        if "task_time" in line:
                            task_time = line.replace("task_time:", "")
                    camera.main_logger.debug(f"task_type:{task_type}, amr_pos_x:{amr_pos_x}, amr_pos_y:{amr_pos_y}, amr_pos_z:{amr_pos_z}, amr_pos_theta:{amr_pos_theta}, amr_tag_id:{amr_tag_id}, ftp_url:{ftp_url}, stitch_state:{stitch_state}, requestor:{requestor}, task_time:{task_time}")        
                    
                    # write to mysql
                    query="""
    insert into `task_history` (`task_type`, `amr_pos_X`, `amr_pos_y`, `amr_pos_z`, 
    `amr_pos_theta`, `amr_tag_id`, `ftp_url`, `task_time`, `stitch_state`, `requestor`) values 
    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s )
    """
                    camera.main_logger.debug(f"query:{query}")    
                    data = (task_type, amr_pos_x, amr_pos_y, amr_pos_z, amr_pos_theta, amr_tag_id, ftp_url, task_time, stitch_state, requestor)
                    
                    if camera.mysql_conn.UpdateRowsByTuple(query, data):
                        # 上傳成功刪圖像
                        camera.main_logger.debug(f"mysql insert successfully!")
                        camera.main_logger.debug("ftp remove imgs")
                        ftp_remove_imgs(camera, pos, task)
                    else:
                        break    


app = Flask(__name__)
# set log level


log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
log.disabled = True


# config
config_obj = ClsConfigParser("./config/config.ini")

# camera
rtsp_url = config_obj.get_config_data("camera", "rtsp_url")
onvif_ip = config_obj.get_config_data("camera", "onvif_ip")
onvif_port = int(config_obj.get_config_data("camera", "onvif_port"))
onvif_account = config_obj.get_config_data("camera", "onvif_account")
onvif_password = config_obj.get_config_data("camera", "onvif_password")
camera = Webcam(filename=rtsp_url,
                apiPreference=cv2.CAP_FFMPEG, device_name="VC-TR30")
camera.onvif = clsONVIFCamera(
    onvif_ip, onvif_port, onvif_account, onvif_password)  # onvif
camera.config = config_obj

# camera offset
camera.camera_offset = int(config_obj.get_config_data("camera", "camera_offset"))

# front camera
front_camera = Webcam(filename="rtsp://192.168.0.22:8553/h264",
                apiPreference=cv2.CAP_FFMPEG, device_name="front-camera")
camera.front_camera = front_camera

# ir cam
camera.ir_cam = FLIRA400()
camera.ir_cam.open_camera()

# logger
main_logger = LogWriter("main")
camera.main_logger = main_logger

# amr
ams_net_id = config_obj.get_config_data("amr", "ams_net_id")
camera.amr = AMR(ams_net_id)

camera.task_requestor = "manual" # task requestor
camera.crr_pan = None
camera.crr_tilt = None
camera.crr_zoom = None
# position folder
camera.pos_folder_x = camera.amr.amr_pos_x
camera.pos_folder_y = camera.amr.amr_pos_y
camera.pos_folder_z = camera.amr.amr_pos_z
camera.pos_folder_theta = camera.amr.amr_pos_theta
camera.pos_folder_tag_id = camera.amr.amr_tag_id
camera.pos_folder = f"({camera.pos_folder_x},{camera.pos_folder_y},{camera.pos_folder_theta},{camera.pos_folder_tag_id})"
camera.task_folder = datetime.now().strftime("%Y%m%d%H%M%S")  # task folder

# initialize
camera.initial_task = Queue() # initialize camera
camera.initial_task.is_running = False
camera.initial_task.thread = Thread()

# panorama
camera.panorama_task = Queue()  # panorama task
camera.panorama_task.pan_list = eval(config_obj.get_config_data(
    "panorama", "pan_list"))  # panorama pan list
camera.panorama_task.tilt_list = eval(config_obj.get_config_data(
    "panorama", "tilt_list"))  # panorama tilt list
camera.panorama_task.zoom_list = eval(config_obj.get_config_data(
    "panorama", "zoom_list"))  # panorama zoom list
camera.panorama_task.is_running = False
camera.panorama_task.stop_flag = False
camera.panorama_task.thread = Thread()
camera.panorama_task.start_time = datetime.now()  # 開始執行時間

# target
camera.target_task = Queue()
camera.target_task.pan_list = eval(config_obj.get_config_data(
    "target", "pan_list"))  # target 拍8張
camera.target_task.tilt_list = eval(config_obj.get_config_data(
    "target", "tilt_list"))  # 一個仰角
camera.target_task.zoom_list = eval(config_obj.get_config_data(
    "target", "zoom_list"))
camera.target_task.is_running = False
camera.target_task.stop_flag = False
camera.target_task.thread = Thread()
camera.target_task.start_time = datetime.now()  # 開始執行時間
camera.target_task.target_pan = 0.0  # 單次目標位置量測
camera.target_task.target_tilt = 0.0
camera.target_task.target_zoom = 0.0

# designated
camera.designated_task = Queue()
camera.designated_task.mysql_host = config_obj.get_config_data(
    "designated", "mysql_host")
camera.designated_task.mysql_user = config_obj.get_config_data(
    "designated", "mysql_user")
camera.designated_task.mysql_password = config_obj.get_config_data(
    "designated", "mysql_password")
camera.designated_task.mysql_database = config_obj.get_config_data(
    "designated", "mysql_database")
# camera.designated_task.mysql_conn = clsMySqlDB(
#     camera.designated_task.mysql_host,
#     camera.designated_task.mysql_user,
#     camera.designated_task.mysql_password,
#     camera.designated_task.mysql_database
# )
camera.designated_task.pan_list = [170.0, 0.0, -170.0]  # target 拍8張
camera.designated_task.tilt_list = [0.0]  # 一個仰角
camera.designated_task.zoom_list = [0.0]
camera.designated_task.is_running = False
camera.designated_task.stop_flag = False
camera.designated_task.thread = Thread()
camera.designated_task.start_time = datetime.now()  # 開始執行時間
camera.designated_task.designated_pan = 0.0
camera.designated_task.designated_tilt = 0.0
camera.designated_task.designated_zoom = 0.0

# ir
camera.ir_task = Queue()
camera.ir_task.is_running = False
camera.ir_task.stop_flag = False
camera.ir_task.thread = Thread()
camera.ir_task.start_time = datetime.now()  # 開始執行時間

# video 
camera.video_task = Queue()
camera.video_task.is_running = False
camera.video_task.stop_flag = False
camera.video_task.thread = Thread()
camera.video_task.start_time = datetime.now() # 開始執行時間

# mysql 
mysql_host = config_obj.get_config_data(
    "mysql", "mysql_host")
mysql_user = config_obj.get_config_data(
    "mysql", "mysql_user")
mysql_password = config_obj.get_config_data(
    "mysql", "mysql_password")
mysql_database = config_obj.get_config_data(
    "mysql", "mysql_database")
camera.mysql_conn = clsMySqlDB(
    mysql_host,
    mysql_user,
    mysql_password,
    mysql_database
)

# ftp
camera.ftp = MyFTP()
camera.ftp.ftp_ip = config_obj.get_config_data("ftp", "ftp_ip")
camera.ftp.ftp_port = int(config_obj.get_config_data("ftp", "ftp_port"))
camera.ftp.ftp_account = config_obj.get_config_data("ftp", "ftp_account")
camera.ftp.ftp_password = config_obj.get_config_data("ftp", "ftp_password")

# img
camera.save_global_coordinate = eval(config_obj.get_config_data(
    "img", "save_global_coordinate"))

def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response
# 設置CORS的應用程式攔截器
app.after_request(add_cors_headers)

@app.route("/")
def index():
    return render_template("index.html")


# web
@app.route("/web/camera_control_web", methods=["GET", "POST"])
def camera_control_web():
    """相機網頁控制"""
    ptz = {
        "pan": [pan for pan in range(-170, 170, 1)],
        "tilt": [tilt for tilt in range(-30, 90, 1)],
        "zoom": [round(0.1*zoom, 1) for zoom in range(0, 11, 1)]
    }
    # 渲染模板
    return render_template("camera_control_web.html", camera=camera, ptz=ptz)


@app.route("/web/amr_control_web", methods=["GET", "POST"])
def amr_control_web():
    """amr control web"""
    # 渲染模板
    return render_template("amr_control_web.html", camera=camera)

@app.route("/web/map", methods=["GET", "POST"])
def map():
    """amr control map"""
    # 渲染模板
    return render_template("map.html", camera=camera)

@app.route("/web/amr_manual_move_panel", methods=["GET", "POST"])
def amr_manual_move_panel():
    """amr manual move panel"""
    # 渲染模板
    return render_template("amr_manual_move_panel.html", camera=camera)


# camera
@app.route("/camera/open_camera", methods=["GET", "POST"])
def open_camera():
    """開啟相機"""
    status, message = False, ""

    # onvif連線檢查
    if not camera.onvif.is_connected():
        if not camera.onvif.open_camera():
            message = "connect to camera failed!"
            data = {"status": status, "message": message}
            return jsonify(data)

    if camera.is_running():
        # 相機已開啟
        message = "camera is already opened!"
        data = {"status": status, "message": message}
        return jsonify(data)

    start = time.time()
    camera.open_camera(background_running=True)  # 開相機
    camera.mysql_conn.Open()  # 開啟mysql連線
    if time.time() - start < 5:
        status = True
        message = "open camera successfully!"
    data = {"status": status, "message": message}
    return jsonify(data)


@app.route("/camera/close_camera", methods=["GET", "POST"])
def close_camera():
    """關閉相機"""
    status, message = False, ""

    if not camera.is_running():
        # 相機已關閉
        message = "camera is already closed!"
        data = {"status": status, "message": message}
        return jsonify(data)

    if camera.panorama_task.is_running:
        # 全景拍攝中
        message = "panorama task is running!"
        data = {"status": status, "message": message}
        return jsonify(data)

    if camera.target_task.is_running:
        # Target候攝中
        message = "target task is running!"
        data = {"status": status, "message": message}
        return jsonify(data)

    camera.onvif.close_camera()  # 關閉onvif
    # camera.amr.close()  # 關閉amr連線
    camera.mysql_conn.Close()  # 開啟mysql連線
    camera.close_camera()  # 關閉相機
    status, message = True, "close camera successfully!"
    data = {"status": status, "message": message}
    return jsonify(data)


@app.route("/camera/get_camera_status", methods=["GET", "POST"])
def get_camera_status():
    """相機狀態"""
    # message = {"is_running": camera.is_running()}
    # data = {"status": True, "message": message}
    # return jsonify(data)
    
    amr = {"is_connected": camera.amr.is_connected(), "amr_pos_x": camera.amr.amr_pos_x, "amr_pos_y": camera.amr.amr_pos_y,
               "amr_pos_theta": camera.amr.amr_pos_theta, "amr_tag_id": camera.amr.amr_tag_id,
               "amr_pos_z": camera.amr.amr_pos_z}
    ptz = {"crr_pan":camera.crr_pan, "crr_tilt":camera.crr_tilt, "crr_zoom":camera.crr_zoom}
    task = {"ir_task":camera.ir_task.is_running, "target_task":camera.target_task.is_running, "designated_task":camera.designated_task.is_running, "inital_task":camera.initial_task.is_running, "video_task":camera.video_task.is_running}
    ir = {"max_temperature":f"{camera.ir_cam.max_temperature_float:.2f}"}
    data = {"status": True, "amr": amr, "ptz":ptz, "task":task, "ir":ir}
    return jsonify(data)


@app.route("/camera/get_camera_img", methods=["GET", "POST"])
def get_camera_img():
    """取即時圖像"""
    data = None
    if camera.is_running():
        img = camera.get_img()
        ret, encoded_img = cv2.imencode(".jpg", img)
        if ret:
            data = encoded_img.tobytes()
    return data


@app.route("/camera/get_camera_base64_img", methods=["GET", "POST"])
def get_camera_base64_img():
    """取即時Base64圖像"""
    return camera.get_base64_img()


@app.route("/camera/move_camera_to_abs/", methods=["GET", "POST"])
def move_camera_to_abs():
    """move camera abs move"""
    camera.main_logger.debug("move camera abs move!")
    success = False
    pan, tilt, zoom = "", "", ""
    if request.args:
        camera.main_logger.debug(f"args:{request.args}")
        try:
            pan = float(request.args.get("pan"))
            tilt = float(request.args.get("tilt"))
            zoom = float(request.args.get("zoom"))
        except Exception as e:
            camera.main_logger.error(f"pass args failed, error:{e}")
        else:
            camera.main_logger.debug(f"send onvif abs_move command!")
            try:
                ptz_angle = [pan, tilt, zoom]
                ptz_value = calculate_ptz_value(ptz_angle)
                ret = camera.onvif.abs_move(ptz_value[0], ptz_value[1], ptz_value[2])
            except Exception as e:
                camera.main_logger.error(f"send onvif failed, error:{e}")
            else:
                success = ret
    data = {"status": success, "pan":pan, "tilt":tilt, "zoom":zoom} 
    return jsonify(data)


# amr
@app.route("/amr/set_amr_pos_x/<int:amr_pos_x>", methods=["GET", "POST"])
def set_amr_pos_x(amr_pos_x):
    """設定AMR X 位置"""
    status = False
    if isinstance(amr_pos_x, int):
        camera.amr.amr_pos_x = amr_pos_x
        status = True
    data = {"status": status, "amr_pos_x": camera.amr.amr_pos_x}
    return jsonify(data)


@app.route("/amr/set_amr_pos_y/<int:amr_pos_y>", methods=["GET", "POST"])
def set_amr_pos_y(amr_pos_y):
    """設定AMR Y 位置"""
    status = False
    if isinstance(amr_pos_y, int):
        camera.amr.amr_pos_y = amr_pos_y
        status = True
    data = {"status": status, "amr_pos_y": camera.amr.amr_pos_y}
    return jsonify(data)


@app.route("/amr/set_amr_pos_theta/<int:amr_pos_theta>", methods=["GET", "POST"])
def set_amr_pos_theta(amr_pos_theta):
    """設定AMR Theta 位置"""
    status = False
    if isinstance(amr_pos_theta, int):
        camera.amr.amr_pos_theta = amr_pos_theta
        status = True
    data = {"status": status, "amr_pos_theta": camera.amr.amr_pos_theta}
    return jsonify(data)


@app.route("/amr/set_amr_tag_id/<int:amr_tag_id>", methods=["GET", "POST"])
def set_amr_tag_id(amr_tag_id):
    """設定AMR Theta 位置"""
    status = False
    if isinstance(amr_tag_id, int):
        camera.amr.amr_tag_id = amr_tag_id
        status = True
    data = {"status": status, "amr_tag_id": camera.amr.amr_tag_id}
    return jsonify(data)


@app.route("/amr/get_amr_status", methods=["GET", "POST"])
def get_amr_status():
    """獲取 AMR 狀態"""
    status = True
    message = {"is_connected": camera.amr.is_connected(), "amr_pos_x": camera.amr.amr_pos_x, "amr_pos_y": camera.amr.amr_pos_y,
               "amr_pos_theta": camera.amr.amr_pos_theta, "amr_tag_id": camera.amr.amr_tag_id,
               "amr_pos_z": camera.amr.amr_pos_z}
    data = {"status": status, "message": message}
    return jsonify(data)


@app.route("/amr/move_amr/", methods=["GET", "POST"])
def move_amr():
    """move amr"""
    camera.main_logger.debug("move amr!")
    status, cmd = False, ""
    if request.args:
        camera.main_logger.debug(f"args:{request.args}")
        try:
            cmd = request.args.get("cmd")
            ret = camera.amr.move_amr(cmd)
        except Exception as e:
            camera.main_logger.error(f"move amr failed, error:{e}")
        else:
            status = ret
            # status = True
    camera.main_logger.debug(f"move amr status:{status}")
    data = {"status": status, "cmd":cmd}
    return jsonify(data)


@app.route("/amr/download_and_upload_obstacle_imgs/", methods=["GET", "POST"])
def download_and_upload_obstacle_imgs():
    """download and upload obstacle_imgs"""
    camera.main_logger.debug("download and upload obstacle_imgs!")
    status, message = False, ""
    try:
        camera.amr.download_and_upload_obstacle_imgs()
    except Exception as e:
        camera.main_logger.error(f"download and upload obstacle imgs failed, error:{e}")
        message = e
    else:
        status = True
    camera.main_logger.debug(f"move amr status:{status}")
    data = {"status": status, "message":message}
    return jsonify(data)


# initial
@app.route("/initial/start_initial_task", methods=["GET", "POST"])
def start_initial_task():
    """Initialize Camera"""
    status, message = False, ""

    if not camera.is_running():
        # 相機未開啟
        message = "camera is closed!"
        data = {"status": status, "message": message}
        return jsonify(data)

    # clear inital task 
    if not camera.initial_task.empty():
        camera.initial_task.queue.clear()
    # add initial task
    camera.initial_task.put((-170.0, 90.0, 0.0 )) # left up
    camera.initial_task.put((170.0, -30.0, 0.0 )) # right down
    camera.initial_task.thread = Thread(target=run_initial_task,
                                         args=(camera,))
    camera.initial_task.thread.start()
    status, message = True, "starting initial task!"
    data = {"status": status, "message": message}
    return jsonify(data)


# panorama
@app.route("/panorama/start_panorama_task", methods=["GET", "POST"])
def start_panorama_task():
    """開始拍攝全景圖工作"""
    camera.main_logger.debug("start panorama task!")

    status, message = False, ""

    if not camera.is_running():
        # 相機未開啟
        message = "camera is closed!"
        data = {"status": status, "message": message}
        return jsonify(data)
    
    requestor = "manual"
    if request.is_json:
        data = request.json
        if data.get("requestor") != "":
            requestor = data.get("requestor")
    camera.main_logger.debug(f"requstor:{requestor}")

    # clear inital task 
    if not camera.initial_task.empty():
        camera.initial_task.queue.clear()
    # add initial task
    camera.initial_task.put((-170.0, 90.0, 0.0 )) # left up
    camera.initial_task.put((170.0, -30.0, 0.0 )) # right down
    
    # 清空task
    if not camera.panorama_task.empty():
        camera.panorama_task.queue.clear()
    # 加入task
    for tilt in camera.panorama_task.tilt_list:
        for pan in camera.panorama_task.pan_list:
            for zoom in camera.panorama_task.zoom_list:
                camera.panorama_task.put(
                    (pan, tilt, zoom))
    # 開始全景拍攝
    camera.task_requestor = requestor
    camera.panorama_task.start_time = datetime.now()  # 開始時間
    camera.pos_folder_x = camera.amr.amr_pos_x
    camera.pos_folder_y = camera.amr.amr_pos_y
    camera.pos_folder_z = camera.amr.amr_pos_z
    camera.pos_folder_theta = camera.amr.amr_pos_theta
    camera.pos_folder_tag_id = camera.amr.amr_tag_id
    camera.pos_folder = f"({camera.pos_folder_x},{camera.pos_folder_y},{camera.pos_folder_theta},{camera.pos_folder_tag_id})"
    camera.task_folder = datetime.now().strftime("%Y%m%d%H%M%S")  # task folder
    camera.panorama_task.thread = Thread(target=run_panorama_task,
                                         args=(camera,))
    camera.panorama_task.thread.start()
    status, message = True, "starting panorama task!"
    data = {"status": status, "message": message}
    return jsonify(data)


@app.route("/panorama/stop_panorama_task", methods=["GET", "POST"])
def stop_panorama_task():
    """結束拍攝全景圖工作"""
    status, message = False, ""

    if not camera.panorama_task.is_running:
        # Panorama Task 未執行
        message = "panorama task is not running!"
        data = {"status": status, "message": message}
        return jsonify(data)

    camera.panorama_task.stop_flag = True
    if camera.panorama_task.thread.is_alive():
        camera.panorama_task.thread.join(timeout=5)
    status, message = True, "panorama task stopped!"
    data = {"status": status, "message": message}
    return jsonify(data)


@app.route("/panorama/get_panorama_task_status/", methods=["GET", "POST"])
def get_panorama_task_status():
    """查詢全景圖工作狀態"""
    status = True
    message = {
        "is_running": camera.panorama_task.is_running,
        "pan_list": camera.panorama_task.pan_list,
        "tilt_list": camera.panorama_task.tilt_list,
        "zoom_list": camera.panorama_task.zoom_list
    }
    data = {"status": status, "message": message}
    return jsonify(data)


@app.route("/panorama/set_panorama_pan_list/", methods=["GET", "POST"])
def set_panorama_pan_list():
    """設定全景圖 PAN 列表"""
    status, message = False, ""

    if camera.panorama_task.is_running:
        message = "panorama task is running!"
        data = {"status": status, "message": message}
        return jsonify(data)

    if camera.target_task.is_running:
        message = "target task is running!"
        data = {"status": status, "message": message}
        return jsonify(data)

    pan_list = request.args.get("pan_list")
    if pan_list:
        camera.panorama_task.pan_list = eval(pan_list)
        status = True
        config_obj.set_config_data(
            "panorama", "pan_list", str(camera.panorama_task.pan_list))
    data = {"status": status, "pan_list": camera.panorama_task.pan_list}
    return jsonify(data)


@app.route("/panorama/get_panorama_pan_list/", methods=["GET", "POST"])
def get_panorama_pan_list():
    """查詢全景圖 PAN 列表"""
    data = {"status": camera.panorama_task.pan_list}
    return jsonify(data)


@app.route("/panorama/set_panorama_tilt_list/", methods=["GET", "POST"])
def set_panorama_tilt_list():
    """設定全景圖 TILT 列表"""
    status, message = False, ""

    if camera.panorama_task.is_running:
        message = "panorama task is running!"
        data = {"status": status, "message": message}
        return jsonify(data)

    if camera.target_task.is_running:
        message = "target task is running!"
        data = {"status": status, "message": message}
        return jsonify(data)

    tilt_list = request.args.get("tilt_list")
    if tilt_list:
        camera.panorama_task.tilt_list = eval(tilt_list)
        status = True
        config_obj.set_config_data(
            "panorama", "tilt_list", str(camera.panorama_task.tilt_list))
    data = {"status": status, "tilt_list": camera.panorama_task.tilt_list}
    return jsonify(data)


@app.route("/panorama/get_panorama_tilt_list/", methods=["GET", "POST"])
def get_panorama_tilt_list():
    """查詢全景圖 TILT 列表"""
    data = {"status": camera.panorama_task.tilt_list}
    return jsonify(data)


@app.route("/panorama/set_panorama_zoom_list/", methods=["GET", "POST"])
def set_panorama_zoom_list():
    """設定全景圖 ZOOM 列表"""
    status, message = False, ""

    if camera.panorama_task.is_running:
        message = "panorama task is running!"
        data = {"status": status, "message": message}
        return jsonify(data)

    if camera.target_task.is_running:
        message = "target task is running!"
        data = {"status": status, "message": message}
        return jsonify(data)

    zoom_list = request.args.get("zoom_list")
    if zoom_list:
        camera.panorama_task.zoom_list = eval(zoom_list)
        status = True
        config_obj.set_config_data(
            "panorama", "zoom_list", str(camera.panorama_task.zoom_list))
    data = {"status": status, "zoom_list": camera.panorama_task.zoom_list}
    return jsonify(data)


@app.route("/panorama/get_panorama_zoom_list/", methods=["GET", "POST"])
def get_panorama_zoom_list():
    """查詢全景圖 ZOOM 列表"""
    data = {"status": camera.panorama_task.zoom_list}
    return jsonify(data)


@app.route("/panorama/get_panorama_ptz_list/", methods=["GET", "POST"])
def get_panorama_ptz_list():
    """查詢全景圖 PTZ 列表"""
    data = {"pan_list": camera.panorama_task.pan_list, "tilt_list": camera.panorama_task.tilt_list,
            "zoom_list": camera.panorama_task.zoom_list}
    return jsonify(data)


# target
@app.route("/target/start_target_task", methods=["GET", "POST"])
def start_target_task():
    """開始拍攝Target環景圖工作"""
    camera.main_logger.debug("start target task!")

    status, message = False, ""

    if not camera.is_running():
        # 相機未開啟
        message = "camera is closed!"
        data = {"status": status, "message": message}
        return jsonify(data)
        
    requestor = "manual"
    if request.is_json:
        data = request.json
        if data.get("requestor") != "":
            requestor = data.get("requestor")
    camera.main_logger.debug(f"requstor:{requestor}")
    
    # clear inital task 
    if not camera.initial_task.empty():
        camera.initial_task.queue.clear()
    # add initial task
    camera.initial_task.put((-170.0, 90.0, 0.0 )) # left up
    camera.initial_task.put((170.0, -30.0, 0.0 )) # right down
    
    # 清空task
    if not camera.target_task.empty():
        camera.target_task.queue.clear()
    # 加入task
    for tilt in camera.target_task.tilt_list:
        for pan in camera.target_task.pan_list:
            for zoom in camera.target_task.zoom_list:
                camera.target_task.put((pan, tilt, zoom))
    # 開始Target全景拍攝
    camera.task_requestor = requestor
    camera.target_task.start_time = datetime.now()  # 開始時間
    camera.pos_folder_x = camera.amr.amr_pos_x
    camera.pos_folder_y = camera.amr.amr_pos_y
    camera.pos_folder_z = camera.amr.amr_pos_z
    camera.pos_folder_theta = camera.amr.amr_pos_theta
    camera.pos_folder_tag_id = camera.amr.amr_tag_id
    camera.pos_folder = f"({camera.pos_folder_x},{camera.pos_folder_y},{camera.pos_folder_theta},{camera.pos_folder_tag_id})"
    camera.task_folder = datetime.now().strftime("%Y%m%d%H%M%S")  # task folder
    camera.target_task.thread = Thread(target=run_target_task,
                                       args=(camera,))
    camera.target_task.thread.start()
    status, message = True, "starting target task!"
    data = {"status": status, "message": message}
    return jsonify(data)


@app.route("/target/stop_target_task", methods=["GET", "POST"])
def stop_target_task():
    """結束拍攝Target環景圖工作"""
    status, message = False, ""

    if not camera.target_task.is_running:
        # Target Task 未執行
        message = "target task is not running!"
        data = {"status": status, "message": message}
        return jsonify(data)

    camera.target_task.stop_flag = True
    if camera.target_task.thread.is_alive():
        camera.target_task.thread.join(timeout=5)
    status, message = True, "target task stopped!"
    data = {"status": status, "message": message}
    return jsonify(data)


@app.route("/target/get_target_task_status/", methods=["GET", "POST"])
def get_target_task_status():
    """查詢Target環景圖工作狀態"""
    status = True
    message = {
        "is_running": camera.target_task.is_running,
        "pan_list": camera.target_task.pan_list,
        "tilt_list": camera.target_task.tilt_list,
        "zoom_list": camera.target_task.zoom_list,
        "target_pan": camera.target_task.target_pan,
        "target_tilt": camera.target_task.target_tilt,
        "target_zoom": camera.target_task.target_zoom
    }
    data = {"status": status, "message": message}
    return jsonify(data)


@app.route("/target/set_target_pan_list/", methods=["GET", "POST"])
def set_target_pan_list():
    """設定Target環景圖 PAN 列表"""
    status, message = False, ""

    if camera.panorama_task.is_running:
        message = "panorama task is running!"
        data = {"status": status, "message": message}
        return jsonify(data)

    if camera.target_task.is_running:
        message = "target task is running!"
        data = {"status": status, "message": message}
        return jsonify(data)

    pan_list = request.args.get("pan_list")
    if pan_list:
        camera.target_task.pan_list = eval(pan_list)
        status = True
        config_obj.set_config_data(
            "target", "pan_list", str(camera.target_task.pan_list))
    data = {"status": status, "pan_list": camera.target_task.pan_list}
    return jsonify(data)


@app.route("/target/get_target_pan_list/", methods=["GET", "POST"])
def get_target_pan_list():
    """查詢Target環景圖 PAN 列表"""
    data = {"status": camera.target_task.pan_list}
    return jsonify(data)


@app.route("/target/set_target_tilt_list/", methods=["GET", "POST"])
def set_target_tilt_list():
    """設定Target環景圖 TILT 列表"""
    status, message = False, ""

    if camera.panorama_task.is_running:
        message = "panorama task is running!"
        data = {"status": status, "message": message}
        return jsonify(data)

    if camera.target_task.is_running:
        message = "target task is running!"
        data = {"status": status, "message": message}
        return jsonify(data)

    tilt_list = request.args.get("tilt_list")
    if tilt_list:
        camera.target_task.tilt_list = eval(tilt_list)
        status = True
        config_obj.set_config_data(
            "target", "tilt_list", str(camera.target_task.tilt_list))
    data = {"status": status, "tilt_list": camera.target_task.tilt_list}
    return jsonify(data)


@app.route("/target/get_target_tilt_list/", methods=["GET", "POST"])
def get_target_tilt_list():
    """查詢Target環景圖 TILT 列表"""
    data = {"status": camera.target_task.tilt_list}
    return jsonify(data)


@app.route("/target/set_target_zoom_list/", methods=["GET", "POST"])
def set_target_zoom_list():
    """設定Target環景圖 ZOOM 列表"""
    status, message = False, ""

    if camera.panorama_task.is_running:
        message = "panorama task is running!"
        data = {"status": status, "message": message}
        return jsonify(data)

    if camera.target_task.is_running:
        message = "target task is running!"
        data = {"status": status, "message": message}
        return jsonify(data)

    zoom_list = request.args.get("zoom_list")
    if zoom_list:
        camera.target_task.zoom_list = eval(zoom_list)
        status = True
        config_obj.set_config_data(
            "target", "zoom_list", str(camera.target_task.zoom_list))
    data = {"status": status, "zoom_list": camera.target_task.zoom_list}
    return jsonify(data)


@app.route("/target/get_target_zoom_list/", methods=["GET", "POST"])
def get_target_zoom_list():
    """查詢Target環景圖 ZOOM 列表"""
    data = {"status": camera.target_task.zoom_list}
    return jsonify(data)


@app.route("/target/set_target_pan/", methods=["GET", "POST"])
def set_target_pan():
    """設定目標位置PAN值"""
    status = False
    target_pan = request.args.get("pan")
    if target_pan:
        if eval(target_pan) >= -170.0 and eval(target_pan) <= 170.0:
            camera.target_task.target_pan = eval(target_pan)
            status = True
    data = {"status": status, "target_pan": camera.target_task.target_pan}
    return jsonify(data)


@app.route("/target/set_target_tilt/", methods=["GET", "POST"])
def set_target_tilt():
    """設定目標位置TILT值"""
    status = False
    target_tilt = request.args.get("tilt")
    if target_tilt:
        if eval(target_tilt) >= -30.0 and eval(target_tilt) <= 90.0:
            camera.target_task.target_tilt = eval(target_tilt)
            status = True
    data = {"status": status, "target_tilt": camera.target_task.target_tilt}
    return jsonify(data)


@app.route("/target/set_target_zoom/", methods=["GET", "POST"])
def set_target_zoom():
    """設定目標位置ZOOM值"""
    status = False
    target_zoom = request.args.get("zoom")
    if eval(target_zoom) >= 0.0 and eval(target_zoom) <= 1.0:
        camera.target_task.target_zoom = eval(target_zoom)
        status = True
    data = {"status": status, "target_zoom": camera.target_task.target_zoom}
    return jsonify(data)


@app.route("/target/get_target_status/", methods=["GET", "POST"])
def get_target_status():
    """獲取目標位置PAN、TILT、ZOOM值"""
    data = {"target_pan": camera.target_task.target_pan,
            "target_tilt": camera.target_task.target_tilt,
            "target_zoom": camera.target_task.target_zoom,
            }
    return jsonify(data)


@app.route("/target/move_to_target/", methods=["POST"])
def move_to_target():
    """移動至Target PTZ位置"""
    status, message = False, ""
    if not camera.is_running():
        # 相機未開啟
        message = "camera is closed!"
        data = {"status": status, "message": message}
        return jsonify(data)
    
    data = request.json
    # 更新資料
    camera.target_task.target_pan = eval(data.get("target_pan"))
    camera.target_task.target_tilt = eval(data.get("target_tilt"))
    camera.target_task.target_zoom = eval(data.get("target_zoom"))

    status = move_to_abs(camera, [camera.target_task.target_pan,
                         camera.target_task.target_tilt,
                         camera.target_task.target_zoom]
                         )
    status = True
    message = {
        "target_pan": camera.target_task.target_pan,
        "target_tilt": camera.target_task.target_tilt,
        "target_zoom": camera.target_task.target_zoom,
        "crr_pan": camera.crr_pan,
        "crr_tilt": camera.crr_tilt,
        "crr_zoom": camera.crr_zoom,
        "amr_pos_x": camera.amr.amr_pos_x,
        "amr_pos_y": camera.amr.amr_pos_y,
        "amr_pos_z": camera.amr.amr_pos_z,
        "amr_pos_theta": camera.amr.amr_pos_theta,
        "amr_tag_id": camera.amr.amr_tag_id
    }
    data = {"status": status, "message": message}
    return jsonify(data)

    
@app.route("/target/update_click_position/", methods=["POST"])
def update_click_position():
    """移動至Target PTZ位置"""
    status, message = True, "update_click_position"
    ptz_status = camera.onvif.get_ptz_status()
    pan, tilt, zoom = ptz_status.get("pan"), \
    ptz_status.get("tilt"), \
    ptz_status.get("zoom")
    pan, tilt, zoom = calculate_ptz_angle([pan, tilt, zoom])

    data = request.json
    target_pan = int(data.get("target_pan"))
    target_tilt = int(data.get("target_tilt"))
    target_zoom = float(data.get("target_zoom"))

    ret = move_to_abs(camera, [pan+target_pan,tilt+target_tilt,zoom+target_zoom])

    data = {"status": ret, "message": target_pan}
    return jsonify(data)

@app.route("/target/get_target_ptz_list/", methods=["GET", "POST"])
def get_target_ptz_list():
    """獲取Target PTZ List"""
    data = {"pan_list": camera.target_task.pan_list,
            "tilt_list": camera.target_task.tilt_list,
            "zoom_list": camera.target_task.zoom_list}
    return jsonify(data)


# designated
@app.route("/designated/start_designated_task", methods=["GET", "POST"])
def start_designated_task():
    """開始拍攝designated工作"""
    camera.main_logger.debug("start designated task!")

    status, message = False, ""

    if not camera.is_running():
        # 相機未開啟
        message = "camera is closed!"
        data = {"status": status, "message": message}
        return jsonify(data)
    
    requestor = "manual"
    if request.is_json:
        data = request.json
        if data.get("requestor") != "":
            requestor = data.get("requestor")
    camera.main_logger.debug(f"requstor:{requestor}")
    
    # clear inital task 
    if not camera.initial_task.empty():
        camera.initial_task.queue.clear()
    # add initial task
    camera.initial_task.put((-170.0, 90.0, 0.0 )) # left up
    camera.initial_task.put((170.0, -30.0, 0.0 )) # right down

    # 清空task
    if not camera.designated_task.empty():
        camera.designated_task.queue.clear()

    # 取得designated task
    query = """
select `primary_key`, `pan`, `tilt`, `zoom`, `amr_pos_x`, `amr_pos_y`, `amr_pos_z`,
amr_pos_theta, amr_tag_id
from `designated_task`
where 1 = 1
and `amr_tag_id`  = %s 
and `task_type` = 'designated' 
"""
    data = (camera.amr.amr_tag_id)
    # data = (0)

    tasks = camera.mysql_conn.SelectRowsByTuple(query, data)
    camera.main_logger.debug(f"amr_tag_id:{camera.amr.amr_tag_id}, tasks:{tasks}")
    if not tasks:
        message = f"no designated task available in tag id: {camera.amr.amr_tag_id}"
        data = {"status": status, "message": message}
        camera.main_logger.debug(message)
        return jsonify(data)
    
    # 加入task
    for task in tasks:
        # pan, theta mysql record data
        pan = task[1]
        theta = task[7]
        camera.main_logger.info(f"theta:{theta}, camera_offset:{camera.camera_offset}, pan:{pan}")
        
        # calculate the target 
        theta_converted = convert_angle(theta)
        camera_pos = theta_converted + camera.camera_offset
        camera_pos_converted = convert_angle(camera_pos)
        target_pos = camera_pos_converted - pan
        target_pos_converted = convert_angle(target_pos)
        camera.main_logger.info(f"theta_converted:{theta_converted}, camera_pos_converted:{camera_pos_converted}, target_converted:{target_pos_converted}")

        # calculate input pan
        input_theta = camera.amr.amr_pos_theta
        input_theta_converted = convert_angle(input_theta)
        input_camera_pos = input_theta_converted + camera.camera_offset
        input_camera_pos_converted = convert_angle(input_camera_pos)
        input_pan = input_camera_pos_converted - target_pos_converted
        camera.main_logger.info(f"input_theta_converted:{input_theta_converted}, input_camera_pos_converted:{input_camera_pos_converted}, input_pan:{input_pan}")
 
        camera.main_logger.info(f"input pan:{input_pan}, input tilt:{task[2]}, input zoom:{task[3]}")
        if input_pan > 170 :
            # calculate reverse directon
            reverse_input_pan = input_pan - 360 # reverse direction
            camera.main_logger.info(f"reverse input pan:{reverse_input_pan}")
            if reverse_input_pan >= -170 and reverse_input_pan <= 170:
                camera.designated_task.put((reverse_input_pan, task[2], task[3])) 
                camera.main_logger.debug(f"add command ok!")
            else: 
                camera.main_logger.error(f"input pan:{input_pan}, is not a valid angle, pass")
        elif input_pan < -170:
            # calculate reverse directon
            reverse_input_pan = input_pan + 360 # reverse direction
            camera.main_logger.info(f"reverse input pan:{reverse_input_pan}")
            if reverse_input_pan >= -170 and reverse_input_pan <= 170:
                camera.designated_task.put((reverse_input_pan, task[2], task[3])) 
                camera.main_logger.debug(f"add command ok!")
            else: 
                camera.main_logger.error(f"input pan:{input_pan}, is not a valid angle, pass")
        else: # -170 ~ 170 ok
            camera.designated_task.put((input_pan, task[2], task[3]))
            camera.main_logger.debug(f"add command ok!")

    # 開始Designated Task拍攝
    camera.task_requestor = requestor
    camera.designated_task.start_time = datetime.now()  # 開始時間
    camera.pos_folder_x = camera.amr.amr_pos_x
    camera.pos_folder_y = camera.amr.amr_pos_y
    camera.pos_folder_z = camera.amr.amr_pos_z
    camera.pos_folder_theta = camera.amr.amr_pos_theta
    camera.pos_folder_tag_id = camera.amr.amr_tag_id
    camera.pos_folder = f"({camera.pos_folder_x},{camera.pos_folder_y},{camera.pos_folder_theta},{camera.pos_folder_tag_id})"
    camera.task_folder = datetime.now().strftime("%Y%m%d%H%M%S")  # task folder
    camera.designated_task.thread = Thread(target=run_designated_task,
                                           args=(camera,))
    camera.designated_task.thread.start()
    status, message = True, "starting designated task!"
    data = {"status": status, "message": message}
    return jsonify(data)


@app.route("/designated/stop_designated_task", methods=["GET", "POST"])
def stop_designated_task():
    """結束拍攝designated工作"""
    status, message = False, ""

    if not camera.designated_task.is_running:
        # Designated Task 未執行
        message = "designated task is not running!"
        data = {"status": status, "message": message}
        return jsonify(data)

    camera.designated_task.stop_flag = True
    if camera.designated_task.thread.is_alive():
        camera.designated_task.thread.join(timeout=5)
    status, message = True, "designated task stopped!"
    data = {"status": status, "message": message}
    return jsonify(data)


@app.route("/designated/write_to_db/", methods=["POST"])
def write_to_db():
    """量測點位寫入Remote DB"""
    status, message = False, ""

    data = request.json
    # 更新資料
    if data.get("pan") == "":
        message = "could not get pan value"
        data = {"status": status, "message": message}
        return jsonify(data)
    pan = eval(data.get("pan"))

    if data.get("tilt") == "":
        message = "could not get tilt value"
        data = {"status": status, "message": message}
        return jsonify(data)
    tilt = eval(data.get("tilt"))

    if data.get("zoom") == "":
        message = "could not get zoom value"
        data = {"status": status, "message": message}
        return jsonify(data)
    zoom = eval(data.get("zoom"))

    if data.get("pos_x") == "":
        message = "could not get pos_x value"
        data = {"status": status, "message": message}
        return jsonify(data)
    pos_x = eval(data.get("pos_x"))

    if data.get("pos_y") == "":
        message = "could not get pos_y value"
        data = {"status": status, "message": message}
        return jsonify(data)
    pos_y = eval(data.get("pos_y"))

    if data.get("pos_z") == "":
        message = "could not get pos_z value"
        data = {"status": status, "message": message}
        return jsonify(data)
    pos_z = eval(data.get("pos_z"))

    if data.get("pos_theta") == "":
        message = "could not get pos_theta value"
        data = {"status": status, "message": message}
        return jsonify(data)
    pos_theta = eval(data.get("pos_theta"))

    if data.get("tag_id") == "":
        message = "could not get tag_id value"
        data = {"status": status, "message": message}
        return jsonify(data)
    tag_id = eval(data.get("tag_id"))

    if data.get("task_type") == "":
        message = "could not get task type value"
        data = {"status": status, "message": message}
        return jsonify(data)
    task_type = data.get("task_type")

    if data.get("video_time") == "":
        message = "could not get video_time value"
        data = {"status": status, "message": message}
        return jsonify(data)
    video_time = eval(data.get("video_time"))

    # 更新資料庫
    query = """
insert into `designated_task` (`pan`, `tilt`, `zoom`,
`amr_pos_x`, `amr_pos_y`, `amr_pos_z`, `amr_pos_theta`, `amr_tag_id`, `create_time`, `task_type`, `video_time`) values
(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""
    data = (pan, tilt, zoom, pos_x, pos_y, pos_z,
            pos_theta, tag_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), task_type, video_time)

    status = camera.mysql_conn.UpdateRowsByTuple(query, data)
    if not status:
        message = "insert into remote DB failed!"
        data = {"status": status, "message": message}
        camera.mysql_conn.ReOpen()
        return jsonify(data)
    message = "insert into remote DB successfully!"
    data = {"status": status, "message": message}
    return jsonify(data)


# ir
@app.route("/ir/start_ir_task", methods=["GET", "POST"])
def start_ir_task():
    """開始拍攝熱顯像圖工作"""
    camera.main_logger.debug(f"start ir task...")
    
    status, message = False, ""

    if not camera.is_running():
        # 相機未開啟
        message = "camera is closed!"
        data = {"status": status, "message": message}
        return jsonify(data)
    
    requestor = "manual"
    if request.is_json:
        data = request.json
        if data.get("requestor") != "":
            requestor = data.get("requestor")
    camera.main_logger.debug(f"requstor:{requestor}")

    # 開始Target全景拍攝
    camera.task_requestor = requestor
    camera.ir_task.start_time = datetime.now()  # 開始時間
    camera.pos_folder_x = camera.amr.amr_pos_x
    camera.pos_folder_y = camera.amr.amr_pos_y
    camera.pos_folder_z = camera.amr.amr_pos_z
    camera.pos_folder_theta = camera.amr.amr_pos_theta
    camera.pos_folder_tag_id = camera.amr.amr_tag_id
    camera.pos_folder = f"({camera.pos_folder_x},{camera.pos_folder_y},{camera.pos_folder_theta},{camera.pos_folder_tag_id})"
    camera.task_folder = datetime.now().strftime("%Y%m%d%H%M%S")  # task folder
    camera.ir_task.thread = Thread(target=run_ir_task, args=(camera,))
    camera.ir_task.thread.start()
    status, message = True, "starting ir task!"
    data = {"status": status, "message": message}
    return jsonify(data)


@app.route("/ir/get_ir_base64_img", methods=["GET", "POST"])
def get_ir_base64_img():
    """取即時IR Base64圖像"""
    return camera.ir_cam.get_base64_img()


@app.route("/ir/get_ir_base64_colormap_img", methods=["GET", "POST"])
def get_ir_base64_colormap_img():
    """取即時IR Base64 ColorMap圖像"""
    return camera.ir_cam.get_base64_colormap_img(mark_max_temp=True)


# video 
@app.route("/video/start_video_task", methods=["GET", "POST"])
def start_video_task():
    """start video task"""
    camera.main_logger.debug(f"start video task...")

    status, message = False, ""

    if not camera.is_running():
        # 相機未開啟
        message = "camera is closed!"
        data = {"status": status, "message": message}
        return jsonify(data)
    
    requestor = "manual"
    if request.is_json:
        data = request.json
        if data.get("requestor") != "":
            requestor = data.get("requestor")
    camera.main_logger.debug(f"requstor:{requestor}")

    # 清空task
    if not camera.video_task.empty():
        camera.video_task.queue.clear()

    # 取得video task
    query = """
select `primary_key`, `pan`, `tilt`, `zoom`, `amr_pos_x`, `amr_pos_y`, `amr_pos_z`,
`amr_pos_theta`, `amr_tag_id`, `video_time` 
from `designated_task`
where 1 = 1
and `amr_tag_id`  = %s 
and `task_type` = 'video' 
"""
    data = (camera.amr.amr_tag_id)
    # data = (0)

    tasks = camera.mysql_conn.SelectRowsByTuple(query, data)
    camera.main_logger.debug(f"amr_tag_id:{camera.amr.amr_tag_id}, tasks:{tasks}")
    if not tasks:
        message = f"no video task available in tag id: {camera.amr.amr_tag_id}"
        data = {"status": status, "message": message}
        camera.main_logger.debug(message)
        return jsonify(data)

    # 加入task
    for task in tasks:
        # pan, theta mysql record data
        pan = task[1]
        theta = task[7]
        video_time = task[9]
        camera.main_logger.info(f"theta:{theta}, camera_offset:{camera.camera_offset}, pan:{pan}, video_time:{video_time}")
        
        # calculate the target 
        theta_converted = convert_angle(theta)
        camera_pos = theta_converted + camera.camera_offset
        camera_pos_converted = convert_angle(camera_pos)
        target_pos = camera_pos_converted - pan
        target_pos_converted = convert_angle(target_pos)
        camera.main_logger.info(f"theta_converted:{theta_converted}, camera_pos_converted:{camera_pos_converted}, target_converted:{target_pos_converted}")

        # calculate input pan
        input_theta = camera.amr.amr_pos_theta
        input_theta_converted = convert_angle(input_theta)
        input_camera_pos = input_theta_converted + camera.camera_offset
        input_camera_pos_converted = convert_angle(input_camera_pos)
        input_pan = input_camera_pos_converted - target_pos_converted
        camera.main_logger.info(f"input_theta_converted:{input_theta_converted}, input_camera_pos_converted:{input_camera_pos_converted}, input_pan:{input_pan}")
 
        camera.main_logger.info(f"input pan:{input_pan}, input tilt:{task[2]}, input zoom:{task[3]}")
        if input_pan > 170 :
            # calculate reverse directon
            reverse_input_pan = input_pan - 360 # reverse direction
            camera.main_logger.info(f"reverse input pan:{reverse_input_pan}")
            if reverse_input_pan >= -170 and reverse_input_pan <= 170:
                camera.video_task.put((reverse_input_pan, task[2], task[3], video_time)) 
                camera.main_logger.debug(f"add video task ok!")
            else: 
                camera.main_logger.error(f"input pan:{input_pan}, is not a valid angle, pass")
        elif input_pan < -170:
            # calculate reverse directon
            reverse_input_pan = input_pan + 360 # reverse direction
            camera.main_logger.info(f"reverse input pan:{reverse_input_pan}")
            if reverse_input_pan >= -170 and reverse_input_pan <= 170:
                camera.video_task.put((reverse_input_pan, task[2], task[3], video_time)) 
                camera.main_logger.debug(f"add video task ok!")
            else: 
                camera.main_logger.error(f"input pan:{input_pan}, is not a valid angle, pass")
        else: # -170 ~ 170 ok
            camera.video_task.put((input_pan, task[2], task[3], video_time))
            camera.main_logger.debug(f"add video task ok!")

    # 開始 Video 拍攝
    camera.task_requestor = requestor
    camera.video_task.start_time = datetime.now()  # 開始時間
    camera.pos_folder_x = camera.amr.amr_pos_x
    camera.pos_folder_y = camera.amr.amr_pos_y
    camera.pos_folder_z = camera.amr.amr_pos_z
    camera.pos_folder_theta = camera.amr.amr_pos_theta
    camera.pos_folder_tag_id = camera.amr.amr_tag_id
    camera.pos_folder = f"({camera.pos_folder_x},{camera.pos_folder_y},{camera.pos_folder_theta},{camera.pos_folder_tag_id})"
    camera.task_folder = datetime.now().strftime("%Y%m%d%H%M%S")  # task folder
    camera.video_task.thread = Thread(target=run_video_task, args=(camera,))
    camera.video_task.thread.start()
    status, message = True, "starting video task!"
    data = {"status": status, "message": message}
    return jsonify(data)


# ftp
@app.route("/ftp/get_pos_list/", methods=["GET", "POST"])
def get_pos_list():
    """取得拍攝點位清單"""
    data = {"status": os.listdir("./save_imgs")}
    return jsonify(data)


@app.route("/ftp/get_task_list/", methods=["GET", "POST"])
def get_task_list():
    """取得拍攝點位記錄"""
    data = {"status": None}
    pos_id = request.args.get("pos")
    if pos_id:
        dir = "./save_imgs" + os.sep + pos_id
        if os.path.exists(dir):
            data = {"status": os.listdir(dir)}
    return jsonify(data)


@app.route("/ftp/get_img_list/", methods=["GET", "POST"])
def get_img_list():
    """取得圖片清單"""
    data = {"status": None}
    pos_id = request.args.get("pos")
    task_id = request.args.get("task")
    if pos_id and task_id:
        dir = "./save_imgs" + os.sep + pos_id + os.sep + task_id
        if os.path.exists(dir):
            data = {"status": os.listdir(dir)}
    return jsonify(data)


@app.route("/ftp/upload_task_imgs/", methods=["GET", "POST"])
def upload_task_imgs():
    """上傳TASK圖片"""
    status = False
    pos_id = request.args.get("pos")
    task_id = request.args.get("task")
    if pos_id and task_id:
        dir = "./save_imgs" + os.sep + pos_id + os.sep + task_id
        if os.path.exists(dir):
            ftp_upload_imgs(camera, pos_id, task_id)
            status = True
    data = {"status": status}
    return jsonify(data)


@app.route("/ftp/remove_task_imgs/", methods=["GET", "POST"])
def remove_task_imgs():
    """刪除TASK舊圖片"""
    status, message = False, ""
    pos_id = request.args.get("pos")
    task_id = request.args.get("task")
    if pos_id and task_id:
        dir = "./save_imgs" + os.sep + pos_id + os.sep + task_id
        if os.path.exists(dir):
            # 刪檔
            files = os.listdir(dir)
            success1, True
            for file in files:
                try:
                    os.remove(dir + os.sep + file)
                except:
                    print(f"remove file:{file} failed!")
                    success1 = False
                    message = "remove file failed!"

            # 刪task資料夾
            success2 = True
            try:
                os.rmdir(dir)
            except:
                print("remove directory failed!")
                success2 = False
                message = f"remove directory failed!"

            if success1 and success2:
                status = True
        else:
            # 無此檔案
            message = "directory not found!"

    data = {"status": status, "message": message}
    return jsonify(data)


# img streamming
def gen_camera_video(camera):
    while True:
        try:
            frame = camera.get_img(resize_img=True)
            ret, jpeg = cv2.imencode(".jpg", frame)
            yield(b'--frame\r\n' 
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.03)
        except:
            print("some thing error")
            break


def gen_front_camera_video(camera):
    while True:
        try:
            frame = camera.front_camera.get_img(resize_img=True)
            ret, jpeg = cv2.imencode(".jpg", frame)
            yield(b'--frame\r\n' 
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.03)
        except:
            print("some thing error")
            break


def gen_ir_camera_video(camera):
    while True:
        try:
            frame = camera.ir_cam.get_colormap_img(mark_max_temp=True)
            ret, jpeg = cv2.imencode(".jpg", frame)
            yield(b'--frame\r\n' 
                  b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            time.sleep(0.03)
        except:
            print("some thing error")
            break


@app.route("/camera/camera_video_feed")
def camera_video_feed():
    return Response(gen_camera_video(camera), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/front_camera/front_camera_video_feed")
def front_camera_video_feed():
    return Response(gen_front_camera_video(camera), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/ir/ir_camera_video_feed")
def ir_camera_video_feed():
    return Response(gen_ir_camera_video(camera), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":

    # 開相機
    camera.open_camera(True)  # 開相機
    front_camera.open_camera(True)
    # 開ADS連線
    # camera.amr.add_panomara_task(camera.panorama_task)  # 引入panorara task
    # camera.amr.add_target_task(camera.target_task)  # 引入target task
    # camera.amr.add_designated_task(
    #     camera.designated_task)  # 引入designated task
    # camera.amr.add_ir_task(camera.ir_task)  # 引入ir task
    camera.amr.add_camera(camera)  # 引入camera
    camera.amr.open(True)  # 開啟amr連線
    # 開啟mysql連線
    # camera.mysql_conn.Open()

    app.run(host="0.0.0.0", threaded=True, debug=False,
            port=8080)
