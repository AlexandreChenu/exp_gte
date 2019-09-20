#ifndef BEST_FIT_ARCH_
#define BEST_FIT_ARCH_

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/nvp.hpp>
#include <sferes/stat/stat.hpp>
#include <sferes/fit/fitness.hpp>
#include <sferes/stat/best_fit.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include "/git/sferes2/exp/exp_direct_target_encode/fit_behav.hpp"

namespace sferes {
  namespace stat {
    SFERES_STAT(BestFitArch, Stat) {
    public:
      template<typename E>
      void refresh(const E& ea) {
        assert(!ea.pop().empty());
        _best = *std::max_element(ea.pop().begin(), ea.pop().end(), fit::compare_max());


        this->_create_log_file(ea, "bestfit.dat");
        if (ea.dump_enabled())
          (*this->_log_file) << ea.gen() << " " << ea.nb_evals() << " " << _best->fit().value() << std::endl;


        if (_cnt == Params::pop::nb_gen){

          int pop_size = ea.pop().size();

          typedef boost::archive::binary_oarchive oa_t;

          std::cout << "test...save..." << std::endl;

          test_and_save(ea);

          std::cout << "tested...saved..." << std::endl;
        }

        _cnt += 1;
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

      template<typename E>
      void test_and_save(const E& ea){

        int cnt = 0;

        std::cout << "starting test and save" << std::endl;

        const std::string filename = ea.res_dir() + "/dict_models.txt";
        std::ofstream dict_file;
        dict_file.open(filename);

        for( auto it = ea.pop().begin(); it != ea.pop().end(); ++it) {

          Eigen::Vector3d target;
          std::vector<double> targ(3);
          targ = *it->gen().get_target();
          target = {targ[0], targ[1], 0};

          double fit = 0;
          fit = run_simu(*it, target); //simulate and obtain fitness and behavior descriptors

          std::cout << "test unitaire - fitness: " << fit << " behavior descriptor: " << target[0] << " " << target[1] << std::endl;

          dict_file << "final_model_" + std::to_string(cnt) << "  " << fit << "  " << target[0] << "  " << target[1] << "\n"; //save simulation results in dictionary file

          typedef boost::archive::binary_oarchive oa_t;
          const std::string fmodel = ea.res_dir() + "/final_model_" + std::to_string(cnt) + ".bin";
          {
          std::ofstream ofs(fmodel, std::ios::binary);

          if (ofs.fail()){
            std::cout << "wolla ca s'ouvre pas" << std::endl;}

          oa_t oa(ofs);
          //oa << model;
          oa << **it;
          } //save model

          }

        dict_file.close();

        std::cout << std::to_string(cnt) + " models saved" << std::endl;
      }

      template <typename T>
      double run_simu(T & model, Eigen::Vector3d target) { 


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

        return out_fit;
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
    };
  }
}
#endif