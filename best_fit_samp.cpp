#ifndef BEST_FIT_SAMP_
#define BEST_FIT_SAMP_

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
    SFERES_STAT(BestFitSamp, Stat) {
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

            double conf_fit = run_simu(*ea.pop()[i], targ);

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

          int n_samp = 233;

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

              double fit_result = run_simu(*ea.pop()[index_max_model], target);

              if (fit_result > 0){
                fits.push_back(fit_result);}
              else{

              }
            }
          }

          input_file.close();

          if (fits.size() > 0){

            double med = median(fits);

            std::cout << "median obtained is: " << med << std::endl;
            //std::cout << "save fitness" << std::endl;

            _medians.push_back(med);
  	  
  	        _fits = {};

  	        for (int i=0; i< fits.size(); i++) //fill _fits in order to save it in a file later
  		        _fits.push_back(fits[i]);

          }
      }
        _cnt += 1;

      if (_cnt == Params::pop::nb_gen){

        std::cout << "Saving medians" << std::endl;

        std::string filename_out = ea.res_dir() + "/fitness_med_large_arch.txt"; //file containing fitness medians
        std::ofstream out_file; 
        out_file.open(filename_out);

        if (!out_file) { //quick check to see if the file is open
          std::cout << "Unable to open file " << filename_out;
          exit(1);}   // call system to stop

        for (int i = 0; i < _medians.size(); i++){
          out_file << _medians[i] << std::endl;
        }
	
	out_file.close();

	std::cout << "Saving fitness and associated target" << std::endl;

	std::string file_fit_targ = ea.res_dir() + "/fitness_target.txt"; //file containing fitness and associated target
	std::ofstream out_file_fit;
  out_file_fit.open(file_fit_targ);

	if (!out_file_fit) { //quick check to see if the file is open
          std::cout << "Unable to open file " << file_fit_targ;
          exit(1);}   // call system to stop

	for (int i = 0; i < _fits.size(); i++){
          out_file_fit << _fits[i] << " " << _targets(i,0) << " " << _targets(i,1) << std::endl;
        }

        out_file_fit.close();
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
    double run_simu(T & model, Eigen::Vector3d target) { 

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
      int _cnt = 0; //not sure if it is useful
      boost::shared_ptr<Phen> _best;
      int _nbest = 3;
      std::vector<double> _medians;
      Eigen::MatrixXd _targets = Eigen::MatrixXd::Zero(233,2);
      std::vector<double> _fits;
    };
  }
}
#endif
