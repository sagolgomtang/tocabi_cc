#include "tocabi_lib/robot_data.h"
#include "wholebody_functions.h"
#include <random>
#include <cmath>

#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Empty.h>
#include <visualization_msgs/Marker.h>
#include <tocabi_msgs/TaskCommand.h>
#include <tocabi_msgs/positionCommand.h>

#include "onnxruntime_cxx_api.h"
#include <urdf/model.h>
#include <array>
#include <deque>
#include <string>
#include <vector>
#include <cstdint>

#ifdef TOCABI_CC_USE_CASADI
#include <casadi/casadi.hpp>
#endif
class CustomController
{
public:
    CustomController(RobotData &rd);
    Eigen::VectorQd getControl();

    //void taskCommandToCC(TaskCommand tc_);
    
    void computeSlow();
    void computeFast();
    void computePlanner();
    void copyRobotData(RobotData &rd_l);

    RobotData &rd_;
    RobotData rd_cc_;

    void loadOnnX();
    void loadArmOnnX();
    void loadJointLimits();
    void loadCasadiCMM();
    void processNoise();
    void processObservation();
    void processArmObservation();
    void processDiscriminator();
    void feedforwardPolicy();
    void feedforwardArmPolicy();
    void initVariable();
    void publishCommandMarker();
    void publishCommandVector();
    void publishActionRate();
    void updatePace();
    
    void quatToTanNorm(const Eigen::Quaterniond& quaternion, Eigen::Vector3d& tangent, Eigen::Vector3d& normal);
    Eigen::Vector3d mat2euler(Eigen::Matrix3d mat);
    Eigen::Vector3d quatRotateInverse(const Eigen::Quaterniond& q, const Eigen::Vector3d& v);


    /////////////////////////////////// ONNX Runtime by Yongarry ///////////////////////////////////////
    size_t input_number, output_number;
    std::vector<std::string> input_names, output_names;
    std::vector<const char *> input_names_char, output_names_char;
    std::vector<Ort::Value> input_tensors, output_tensors;

    std::vector<std::vector<float>> input_states_buffer;
    std::vector<float> state_cur_, state_buffer_;

    // for long history observation
    std::vector<float> state_long_hist_, state_long_hist_buffer_;

    int input_obs_idx_ = 0;
    int arm_input_obs_idx_ = 0;

    // Arm policy ONNX
    Ort::Session arm_session;
    size_t arm_input_number = 0, arm_output_number = 0;
    std::vector<std::string> arm_input_names, arm_output_names;
    std::vector<const char *> arm_input_names_char, arm_output_names_char;
    std::vector<Ort::Value> arm_input_tensors, arm_output_tensors;
    std::vector<std::vector<float>> arm_input_states_buffer;
    std::vector<float> arm_state_cur_, arm_state_buffer_;

    ///////////////////////////////////// Actor-Critic Network ///////////////////////////////////////
    static const int num_action = 12;
    static const int num_actuator_action = 12;
    static const int num_arm_action = 8;
    // static const int num_cur_state = 49; // 37 + 12
    // LegActor obs: internal state excludes last_action (num_action).
    static const int num_cur_state = 50;
    static const int num_cur_internal_state = num_cur_state - num_action;
    static const int num_state_skip = 2;
    // static const int num_state_skip = 1;
    static const int num_state_hist = 10;
    // static const int num_state_hist = 1;
    static const int num_state = num_cur_internal_state*num_state_hist+num_action*(num_state_hist-1);
    // static const int num_state = num_cur_internal_state +num_action;

    // for long history observation
    static const int num_long_hist_skip = 10;
    static const int num_long_hist_len = 50;
    static const int num_hist_state = num_long_hist_len * num_long_hist_skip;

    // ArmActor obs: 3(base ang vel) + 3(gravity) + 8(q) + 8(qdot) + 8(last) + 3(CAM) + 3(CAM_des)
    static const int num_arm_state = 36;

    Eigen::MatrixXd rl_action_, rl_action_pre_, torq_diff_, energy;
    Eigen::Matrix<double, num_arm_action, 1> rl_action_arm_, rl_action_arm_pre_;
    double value_;
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    bool stop_by_value_thres_ = false;
    bool enable_value_stop_ = false;
    Eigen::Matrix<double, MODEL_DOF, 1> q_stop_;
    float stop_start_time_;

    std::ofstream writeFile;

    bool is_on_robot_ = false;
    bool is_write_file_ = true;
    bool is_hist_encoder_ = false;

    Eigen::Matrix<double, MODEL_DOF, 1> q_dot_lpf_;

    Eigen::Matrix<double, MODEL_DOF, 1> q_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_init_mode7_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_noise_pre_;
    Eigen::Matrix<double, MODEL_DOF, 1> q_vel_noise_, q_vel_noise_pre_;

    Eigen::Matrix<double, MODEL_DOF, 1> torque_init_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_spline_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_rl_;
    Eigen::Matrix<double, MODEL_DOF, 1> torque_bound_;

    Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kp_;
    Eigen::Matrix<double, MODEL_DOF, MODEL_DOF> kv_;

    Eigen::VectorQd Gravity_MJ_;

    Eigen::Vector6d LF_CF_FT_pre, RF_CF_FT_pre = Eigen::Vector6d::Zero();

    float start_time_;
    float time_inference_pre_ = 0.0;
    float time_write_pre_ = 0.0;

    double time_cur_;
    double time_pre_;

    Eigen::Vector3d euler_angle_;
    Eigen::Vector3d tan_vec, nor_vec;

    // float ft_left_init_ = 500.0;
    // float ft_right_init_ = 500.0;

    string weight_dir_ = "";
    string arm_weight_dir_ = "";
    bool use_arm_policy_ = false;
    bool include_cam_obs_ = true;
    bool use_obs_history_layout_ = false;
    bool use_obs_joint_vel_lpf_ = true;
    bool use_dtau_joint_vel_lpf_ = true;
    double policy_hz_ = 50.0;
    bool obs_history_layout_warned_ = false;
    std::array<std::array<double, 2>, num_actuator_action> leg_joint_pos_limits_;
    std::array<std::array<double, 2>, num_arm_action> arm_joint_pos_limits_;
    std::string policy_with_arm_path_;
    std::string policy_without_arm_path_;
    // Joystick
    ros::NodeHandle nh_;

    void joyCallback(const sensor_msgs::Joy::ConstPtr& joy);
    void xBoxJoyCallback(const sensor_msgs::Joy::ConstPtr& joy);
    void keyboardCmdCallback(const std_msgs::Float32MultiArray::ConstPtr& msg);
    void mode7ToggleCallback(const std_msgs::Empty::ConstPtr& msg);
    void handleMode7ToggleRequest();
    void pollDirectJoystick();
    void closeDirectJoystick();
    void directJoystickTimerCallback(const ros::TimerEvent&);
    void simTimeCallback(const std_msgs::Float32ConstPtr& msg);
    ros::Subscriber joy_sub_;
    ros::Subscriber xbox_joy_sub_;
    ros::Subscriber keyboard_cmd_sub_;
    ros::Subscriber mode7_toggle_sub_;
    ros::Publisher sim_command_pub_;
    ros::Subscriber sim_time_sub_;
    ros::Publisher cmd_marker_pub_;
    ros::Publisher cmd_vec_pub_;
    ros::Publisher cam_cmd_pub_;
    ros::Publisher action_rate_pub_;
    ros::Publisher gui_send_pub_;
    ros::Publisher gui_cmd_pub_;
    ros::Subscriber gui_send_sub_;
    ros::Publisher task_cmd_pub_;
    ros::Publisher pos_cmd_pub_;
    ros::Timer direct_joy_timer_;
    bool prev_btn9_ = false;
    bool prev_btn0_ = false;
    bool prev_btn10_ = false;
    bool prev_btn1_ = false;
    bool prev_btn8_ = false;
    bool prev_btn2_ = false;
    bool prev_btn3_ = false;
    bool prev_btn12_ = false;
    bool btn1_gravity_stopped_ = false;
    bool cmd_zero_lock_ = false;
    bool joystick_enabled_ = false;
    bool direct_joystick_enabled_ = false;
    std::string direct_joystick_device_ = "/dev/input/js0";
    int direct_joystick_fd_ = -1;
    double direct_joystick_last_open_try_s_ = -1.0;
    std::vector<float> direct_joy_axes_;
    std::vector<int32_t> direct_joy_buttons_;
    std::vector<int32_t> direct_joy_press_latch_;
    double cmd_stop_min_phase_cycles_ = 0.0;
    int prev_axis6_dir_ = 0;
    void guiSendCallback(const std_msgs::Empty::ConstPtr& msg);
    double cmd_vis_scale_ = 1.0;

    Eigen::Vector3d local_lin_vel_;
    Eigen::Vector3d base_ang_vel_bf_lpf_ = Eigen::Vector3d::Zero();
    bool base_ang_vel_lpf_initialized_ = false;
    int64_t base_ang_vel_lpf_last_us_ = 0;
    Eigen::Vector3d cam_bf_;
    Eigen::Vector3d cam_des_bf_;

    Eigen::Matrix<double, 12, 12> action_offset_, 
                                  action_scale_;
    double target_vel_x_ = 0.0;
    double target_vel_y_ = 0.0;
    double target_vel_yaw_ = 0.0;
    double target_vel_raw_x_ = 0.0;
    double target_vel_raw_y_ = 0.0;
    double target_vel_raw_yaw_ = 0.0;
    double cmd_ema_window_s_ = 0.2;
    int64_t cmd_ema_last_us_ = 0;
    bool cmd_ema_initialized_ = false;

    float desired_vel_x = 0.0;
    float desired_vel_yaw = 0.0;
    bool value_valid_ = false;
    double phase_period_s_ = 1.6;
    double phase_offset_s_ = 0.0;
    bool phase_started_ = false;
    double phase_start_time_s_ = 0.0;
    double phase_time_s_ = 0.0;
    int64_t phase_last_update_us_ = 0;
    bool sim_paused_ = false;
    ros::Time last_sim_time_msg_;
    bool sim_time_received_ = false;
    double sim_time_s_ = 0.0;
    double sim_time_prev_s_ = 0.0;
    double last_sim_time_observed_s_ = -1.0;
    double cmd_scale_x_ = 1.0;
    double cmd_scale_y_ = 0.5;
    double cmd_scale_yaw_ = 0.6;
    double joystick_deadzone_ = 0.1;

    Eigen::Matrix<double, num_action, 1> action_rate_;
    std::deque<std::vector<float>> leg_hist_core_queue_;

    bool pace_trigger_ = false;
    bool pace_active_ = false;
    double pace_start_time_s_ = 0.0;
    double pace_last_pub_time_s_ = 0.0;
    double pace_duration_s_ = 10.0;
    double pace_min_freq_ = 0.1;
    double pace_max_freq_ = 10.0;
    Eigen::Matrix<double, 12, 1> pace_direction_;
    Eigen::Matrix<double, 12, 1> pace_bias_;
    Eigen::Matrix<double, 12, 1> pace_scale_;
    Eigen::VectorQd pace_hold_q_;
    std::vector<double> pace_pos_kp_backup_;
    std::vector<double> pace_pos_kv_backup_;
    bool pace_gains_saved_ = false;
    bool pace_align_active_ = false;
    double pace_align_start_s_ = 0.0;
    double pace_align_duration_s_ = 2.0;
    double pace_align_gain_ = 3.0;
    bool pace_hold_after_ = true;
    bool pace_collect_enabled_ = false;
    bool pace_collect_saved_ = false;
    double pace_collect_start_time_s_ = 0.0;
    std::string pace_collect_output_dir_ = "/home/user/other_ws/src/pace-sim2real/data/tocabi_mujoco";
    std::string pace_collect_output_prefix_ = "chirp_data";
    std::vector<double> pace_collect_time_;
    std::vector<Eigen::VectorQd> pace_collect_q_;
    std::vector<Eigen::VectorQd> pace_collect_des_;
    bool pace_pc_gravity_saved_ = false;
    bool pace_pc_gravity_backup_ = true;
    bool pace_has_prev_des_ = false;
    Eigen::VectorQd pace_prev_des_;

    Eigen::VectorQd q_min_;
    Eigen::VectorQd q_max_;
    bool has_joint_limits_ = false;
    double q_limit_scale_ = 1.0;

    bool use_casadi_cam_ = false;
    std::string casadi_cmm_path_;
// #ifdef TOCABI_CC_USE_CASADI
//     bool casadi_cam_ready_ = false;
// #else
//     bool casadi_cam_ready_ = false;
// #endif
// #ifdef TOCABI_CC_USE_CASADI
//     casadi::Function cmm_fn_;
//     bool casadi_cam_ready_ = false;
// #endif
    int debug_log_steps_remaining_ = 0;
    bool debug_log_this_step_ = false;
    bool mode7_active_ = false;
    bool init_pose_hold_toggle_request_ = false;
    bool init_pose_hold_active_ = false;
    bool mode6_active_ = false;
    bool mode6_done_ = false;
    int prev_tc_mode_ = -1;
    bool mode6_logged_ = false;
    bool mode7_send_triggered_ = false;
    size_t test_log_step_ = 0;
    size_t test_policy_step_ = 0;
    std::string test_log_dir_ = "/home/user/tocabi_mujoco_ws/src/tocabi_cc/test_log";
    std::ofstream test_obs_file_;
    std::ofstream test_act_file_;
    std::ofstream test_act_mapped_file_;
    std::ofstream test_action_rate_stats_file_;
    std::vector<Eigen::Matrix<double, 12, 1>> test_act_buffer_;
    std::string test_act_runtime_name_ = "leg_actions_sim.csv";
    std::string test_act_mapped_name_ = "leg_actions_sim_mapped.csv";
    std::string test_action_rate_stats_name_ = "action_rate_stats_sim.csv";
    bool test_action_rate_stats_last_valid_ = false;
    double test_action_rate_stats_last_mean_abs_ = 0.0;
    double test_action_rate_stats_last_max_abs_ = 0.0;
    double action_rate_timeavg_sum_mean_ = 0.0;
    double action_rate_timeavg_sum_max_ = 0.0;
    size_t action_rate_timeavg_count_ = 0;
    int64_t action_rate_timeavg_last_print_us_ = 0;
    std::string test_obs_input_name_ = "leg_actor_obs.txt";
    std::vector<std::vector<float>> test_obs_buffer_;
    size_t test_obs_idx_ = 0;
    size_t test_max_steps_ = 0;
    std::ofstream test_obs_sim_file_;
    std::string test_obs_sim_name_ = "leg_actor_obs_sim.csv";
    size_t test_obs_sim_step_ = 0;
    bool test_obs_sim_done_ = false;
    size_t test_obs_sim_max_steps_ = 1000;

    bool se_log_active_ = false;
    size_t se_log_step_ = 0;
    std::string se_log_dir_ = "/home/user/tocabi_mujoco_ws/state_estimation_log";
    std::string se_log_name_ = "state_estimation_obs.txt";
    std::ofstream se_log_file_;

private:
    Eigen::VectorQd ControlVal_;

    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;

    const std::string reset = "\033[0m";     // Reset color
    const std::string red = "\033[31m";     // Red
    const std::string green = "\033[32m";   // Green
    const std::string yellow = "\033[33m";  // Yellow
    const std::string blue = "\033[34m"; 
};
