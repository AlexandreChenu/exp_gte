#ifndef BEST_FIT_NOV_
#define BEST_FIT_NOV_

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/nvp.hpp>
#include <sferes/stat/stat.hpp>
#include <sferes/fit/fitness.hpp>
#include <sferes/stat/best_fit.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "fit_behav.hpp"

namespace sferes {
  namespace stat {
    SFERES_STAT(BestFitNov, Stat) {
    public:
      template<typename E>
      void refresh(const E& ea) {
        assert(!ea.pop().empty());
        _best = *std::max_element(ea.pop().begin(), ea.pop().end(), fit::compare_max());


        this->_create_log_file(ea, "bestfit.dat");
        if (ea.dump_enabled())
          (*this->_log_file) << ea.gen() << " " << ea.nb_evals() << " " << _best->fit().value() << std::endl;

        //change it to depend from params 
        if (_cnt%Params::pop::dump_period == 0){ //for each dump period

          std::cout << "size of population: " << ea.pop().size() << std::endl;

          std::vector<double> fits; //vector of fitness for computing median
          //std::vector<auto> its; // vector of iterators
          std::vector<std::vector<double>> known_targets; //vector of targets associated with iterators
          std::vector<int> indx_targets;

          std::vector <double> conf_fits;

          for (int i = 0; i < ea.pop().size(); ++i){

            Eigen::Vector3d targ;
            targ[0] = ea.pop()[i]->gen().get_target()[0];
            targ[1] = ea.pop()[i]->gen().get_target()[1];
            targ[2] = 0;

            double conf_fit = run_simu_fit(*ea.pop()[i], targ);

            if (conf_fit > 0){ //considere target as known only if fitness is higher than 0
              known_targets.push_back(ea.pop()[i]->gen().get_target()); //get target output a target of dimension 3 
              indx_targets.push_back(i);
              conf_fits.push_back(conf_fit);}
          }

          std::cout << "number of known targets: " << known_targets.size() << std::endl;

          std::string filename_in = "/git/sferes2/exp/ex_data/samples_cart.txt"; //file containing samples
          std::ifstream input_file; 
          input_file.open(filename_in);

          if (!input_file) { //quick check to see if the file is open
            std::cout << "Unable to open file " << filename_in;
            exit(1);}   // call system to stop

          int n_samp = 10;
          std::vector<double> sums_zones;
          double mean_sum_zones =0;

          for (int s=0; s < n_samp; s++){


            Eigen::Vector3d target;
            double out; //get sample's target position
            input_file >> out;
            target[0] = out;
            input_file >> out;
            target[1] = out;
            target[2] = 0; 

	          _targets(s,0) = target[0];
	          _targets(s,1) = target[1];

            std::cout << "target is: " << target[0] << " " << target[1] << std::endl;

            double max_fit_dist = -INFINITY;
            int index_max_model = 0;
            int index_max_target = 0;

            if (known_targets.size() > 0){

              std::cout << "look for closest target in archive" << std::endl;

              for (int i=0; i<known_targets.size(); i++){
                  double i_fit_dist = conf_fits[i]/sqrt((known_targets[i][0]- target[0])*(known_targets[i][0]- target[0]) + (known_targets[i][1]- target[1])*(known_targets[i][1]- target[1]));
                  if (i_fit_dist > max_fit_dist){
                    max_fit_dist = i_fit_dist;
                    index_max_model = indx_targets[i]; //index in the population vector of the corresponding model
                    index_max_target = i;
                  }
              }
            
              std::cout << "closest target with best fitness obtained is: " << known_targets[index_max_target][0] << " " << known_targets[index_max_target][1] << std::endl;

              Eigen::MatrixXd zones_cnt_int = Eigen::MatrixXd::Zero(101,101);

              zones_cnt_int = run_simu(*ea.pop()[index_max_model], target);

              int sum_zones = 0;

              for (float i = 0; i < 101; i+=1){
                for (float j = 0; j < 101; j+=1){
                  if (zones_cnt_int(i,j) != 0)
                    sum_zones += 1;
                }}

            sums_zones.push_back(sum_zones);
          
        
          double s =0;
          for(int i=0; i<sums_zones.size(); i++){
            s += sums_zones[i];}

          mean_sum_zones = s/sums_zones.size();
        }}
          double novelty_score = mean_sum_zones;

          std::cout << "novelty score is: " << novelty_score << std::endl;
          _nov_scores.push_back(novelty_score);

          input_file.close();

      }
        _cnt += 1;

      if (_cnt == Params::pop::nb_gen){

        std::cout << "Saving means" << std::endl;

        std::string filename_out = ea.res_dir() + "/novelty_samp.txt"; //file containing fitness medians
        std::ofstream out_file; 
        out_file.open(filename_out);

        if (!out_file) { //quick check to see if the file is open
          std::cout << "Unable to open file " << filename_out;
          exit(1);}   // call system to stop

        for (int i = 0; i < _nov_scores.size(); i++){
          out_file << _nov_scores[i] << std::endl;
        }
	       out_file.close();
        }
      }


      void show(std::ostream& os, size_t k) {
        _best->develop();
        _best->show(os);
        _best->fit().set_mode(fit::mode::view);
        _best->fit().eval(*_best);
      }
      const boost::shared_ptr<Phen> best() const {
        return _best;
      }
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version) {
        ar & BOOST_SERIALIZATION_NVP(_best);
      }
        template <typename T>
    double run_simu_fit(T & model, Eigen::Vector3d target) { 

        //std::cout << "start initialization" << std::endl;

        //init variables
        double _vmax = 1;
        double _delta_t = 0.1;
        double _t_max = 10; //TMax guidé poto
        Eigen::Vector3d robot_angles;

        Eigen::Vector3d prev_pos; //compute previous position
        Eigen::Vector3d pos_init;

        robot_angles = {0,M_PI,M_PI}; //init everytime at the same place

        double radius;
        double theta;

        model.develop();

        double dist = 0;

        //get gripper's position
        prev_pos = forward_model(robot_angles);
        pos_init = forward_model(robot_angles);

        std::vector<float> inputs(5);

        //std::cout << "initialization done" << std::endl;


        for (int t=0; t< _t_max/_delta_t; ++t){
              
              inputs[0] = target[0] - prev_pos[0]; //get side distance to target
              inputs[1] = target[1] - prev_pos[1]; //get front distance to target
              inputs[2] = robot_angles[0];
              inputs[3] = robot_angles[1];
              inputs[4] = robot_angles[2];

              ////DATA GO THROUGH NN
              //model.nn().init(); //init neural network 

              for (int j = 0; j < model.gen().get_depth() + 1; ++j) //In case of FFNN
                model.nn().step(inputs);
              
              Eigen::Vector3d output;
              for (int indx = 0; indx < 3; ++indx){
                output[indx] = 2*(model.nn().get_outf(indx) - 0.5)*_vmax; //Remap to a speed between -v_max and v_max (speed is saturated)
                robot_angles[indx] += output[indx]*_delta_t; //Compute new angles
              }

              //Eigen::Vector3d new_pos;
              prev_pos = forward_model(robot_angles); //remplacer pour ne pas l'appeler deux fois

              if (sqrt(square(target.array() - prev_pos.array()).sum()) < 0.02){
                dist -= sqrt(square(target.array() - prev_pos.array()).sum());}


             else {
                dist -= (log(1+t)) + (sqrt(square(target.array() - prev_pos.array()).sum()));}
            }

        Eigen::Vector3d final_pos; 
        final_pos = forward_model(robot_angles);

        double out_fit;

        if (sqrt(square(target.array() - final_pos.array()).sum()) < 0.02){
          out_fit = 1.0 + dist/500;} // -> 1

        else {
          out_fit = dist/500;} // -> 0

        std::cout << "fitness: " << out_fit << std::endl;
      
        //std::cout << "test done" << std::endl;

        return out_fit;
    }


    template <typename T>
    Eigen::MatrixXd run_simu(T & model, Eigen::Vector3d target) { 

        //std::cout << "start initialization" << std::endl;

        //init variables
        double _vmax = 1;
        double _delta_t = 0.1;
        double _t_max = 10; //TMax guidé poto
        Eigen::Vector3d robot_angles;

        Eigen::MatrixXd work_zones_cnt = Eigen::MatrixXd::Zero(101,101);

        Eigen::Vector3d prev_pos; //compute previous position
        Eigen::Vector3d pos_init;

        robot_angles = {0,M_PI,M_PI}; //init everytime at the same place

        double radius;
        double theta;

        model.develop();

        double dist = 0;

        //get gripper's position
        prev_pos = forward_model(robot_angles);
        pos_init = forward_model(robot_angles);

        std::vector<float> inputs(5);

        //std::cout << "initialization done" << std::endl;


        for (int t=0; t< _t_max/_delta_t; ++t){
              
              inputs[0] = target[0] - prev_pos[0]; //get side distance to target
              inputs[1] = target[1] - prev_pos[1]; //get front distance to target
              inputs[2] = robot_angles[0];
              inputs[3] = robot_angles[1];
              inputs[4] = robot_angles[2];

              ////DATA GO THROUGH NN
              //model.nn().init(); //init neural network 

              for (int j = 0; j < model.gen().get_depth() + 1; ++j) //In case of FFNN
                model.nn().step(inputs);
              
              Eigen::Vector3d output;
              for (int indx = 0; indx < 3; ++indx){
                output[indx] = 2*(model.nn().get_outf(indx) - 0.5)*_vmax; //Remap to a speed between -v_max and v_max (speed is saturated)
                robot_angles[indx] += output[indx]*_delta_t; //Compute new angles
              }

              //Eigen::Vector3d new_pos;
              prev_pos = forward_model(robot_angles); //remplacer pour ne pas l'appeler deux fois

              int x_int = prev_pos[0]*100;
              int y_int = prev_pos[1]*100;

              int indx_X =0;
              int indx_Y =0;
              
              if (x_int %2 !=0)
                  indx_X = (x_int + 100)/2;
              
              else 
                  indx_X = (x_int + 101)/2 ;
              
              if (y_int %2 !=0)
                  indx_Y = (y_int + 100)/2;
              
              else 
                  indx_Y = (y_int + 101)/2;
          
              work_zones_cnt(indx_X,indx_Y) ++;

            }

        Eigen::Vector3d final_pos; 
        final_pos = forward_model(robot_angles);


        if (sqrt(square(target.array() - final_pos.array()).sum()) < 0.02){
          return work_zones_cnt;} // -> 1

        else {
          return Eigen::MatrixXd::Zero(101,101);} // -> 0
    }

    double median(std::vector<double> &v)
    {
      size_t n = v.size() / 2;
      std::nth_element(v.begin(), v.begin()+n, v.end());
      return v[n];
    }

    Eigen::Vector3d forward_model(Eigen::VectorXd a){
    
    Eigen::VectorXd _l_arm=Eigen::VectorXd::Ones(a.size()+1);
    _l_arm(0)=0;
    _l_arm = _l_arm/_l_arm.sum();

    Eigen::Matrix4d mat=Eigen::Matrix4d::Identity(4,4);

    for(size_t i=0;i<a.size();i++){

      Eigen::Matrix4d submat;
      submat<<cos(a(i)), -sin(a(i)), 0, _l_arm(i), sin(a(i)), cos(a(i)), 0 , 0, 0, 0, 1, 0, 0, 0, 0, 1;
      mat=mat*submat;
    }
    
    Eigen::Matrix4d submat;
    submat<<1, 0, 0, _l_arm(a.size()), 0, 1, 0 , 0, 0, 0, 1, 0, 0, 0, 0, 1;
    mat=mat*submat;
    Eigen::VectorXd v=mat*Eigen::Vector4d(0,0,0,1);

    return v.head(3);
 }

     protected:
      boost::shared_ptr<Phen> _best;
      int _cnt = 0; //not sure if it is useful
      Eigen::MatrixXd _targets = Eigen::MatrixXd::Zero(233,2);
      std::vector<double> _nov_scores;
    };
  }
}
#endif


