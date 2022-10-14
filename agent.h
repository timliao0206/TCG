/**
 * Framework for Threes! and its variants (C++ 11)
 * agent.h: Define the behavior of variants of agents including players and environments
 *
 * Author: Theory of Computer Games
 *         Computer Games and Intelligence (CGI) Lab, NYCU, Taiwan
 *         https://cgilab.nctu.edu.tw/
 */

#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include <fstream>
#include "board.h"
#include "action.h"
#include "weight.h"

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

/**
 * base agent for agents with randomness
 */
class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * base agent for agents with weight tables and a learning rate
 */
class weight_agent : public agent {
public:
	weight_agent(const std::string& args = "") : agent(args), alpha(0) {
		if (meta.find("init") != meta.end())
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end())
			load_weights(meta["load"]);
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~weight_agent() {
		if (meta.find("save") != meta.end())
			save_weights(meta["save"]);
	}

protected:
	virtual void init_weights(const std::string& info) {
		std::string res = info; // comma-separated sizes, e.g., "65536,65536"
		for (char& ch : res)
			if (!std::isdigit(ch)) ch = ' ';
		std::stringstream in(res);
		for (size_t size; in >> size; net.emplace_back(size));
	}
	virtual void load_weights(const std::string& path) {
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path) {
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}

protected:
	std::vector<weight> net;
	float alpha;
};

/***
 * n-tuple network feature
 * | 0 | 1 | 2 | 3 |
 * | 4 | 5 | 6 | 7 |
 * | 8 | 9 | 10| 11|
 * | 12| 13| 14| 15|
 * 
 * -1 being the hint tile
 ***/
class Feature{
public:
	Feature(const std::vector<int>& feat){
		feature = feat;
	}

	long long operator() (const board& board) const{
		long long idx = 0LL;

		int empty = 0;

		for(int i=0;i<16;i++){
			if(board(i) == 0) empty++;
		}

		for(int feat : feature){
			idx <<= 4;

			if(feat == -1)
				idx |= empty;// board.hint();
			else
				idx |= board(feat);
		}
		return idx;
	}

	int size() const{
		return feature.size();
	}

	int hash() const{
		int i=0;
		bool flag = 0;
		for(int k : feature){
			if(k == -1){
				flag = 1;
				continue;
			}
			i |= (1 << k);
		}

		return flag ? -i : i;
	}

	const std::vector<int>& get_feature() const{
		return feature;
	}

	bool operator==(const Feature& feat) const{
		return this->hash() == feat.hash();
	}
private:
	std::vector<int> feature;
};

class IsoFeature{
public:

	IsoFeature(){};
	IsoFeature(const Feature& feature){
		create_iso_feature(feature);
	}

	std::vector<int> left_rotate(const std::vector<int>& feat){
		std::vector<int> vec;
		for(int i:feat){
			if(i == -1) continue;
			vec.push_back(_right_rotation_matching[i]);
		}

		return vec;
	}

	std::vector<int> reflection(const std::vector<int>& feat){
		std::vector<int> vec;
		for(int i:feat){
			if(i == -1) continue;
			vec.push_back(_reflection_matching[i]);
		}

		return vec;
	}

	void create_iso_feature(Feature feat){
		std::vector< std::pair<int,Feature> > iso;

		Feature tmp = feat;
		auto vec = feat.get_feature();
		bool with_hint = std::find(vec.begin(),vec.end(),-1) != vec.end();

		for(int i=0;i<5;i++){
			tmp =  Feature( left_rotate( tmp.get_feature() ) );
			iso.push_back(std::make_pair(tmp.hash(),tmp));
		}

		tmp =  Feature( reflection( tmp.get_feature() ) );

		for(int i=0;i<4;i++){
			iso.push_back(std::make_pair(tmp.hash(),tmp));
			tmp =  Feature( left_rotate( tmp.get_feature() ) );
		}

		//remove duplicate
		std::sort(iso.begin(),iso.end(),[](std::pair<int,Feature> a,std::pair<int,Feature> b){return a.first > b.first;});

		iso.erase(std::unique(iso.begin(),iso.end()),iso.end());

		for(auto f : iso){
			auto vec = f.second.get_feature();
			if(with_hint)
				vec.push_back(-1);
			iso_feature.push_back(Feature(vec));
		}

		return;
	}

	const std::vector< Feature >& get_all_feature() const{
		return iso_feature;
	}
private:
	std::vector< Feature > iso_feature;


	std::vector<int> _right_rotation_matching = {12,8,4,0,13,9,5,1,14,10,6,2,15,11,7,3};

	std::vector<int> _reflection_matching = {15,11,7,3,14,10,6,2,13,9,5,1,12,8,4,0};


};

/***
 * weighted n-tuple network
 ***/

const std::vector< std::vector<int> > feats = {
	{0,1,2,3,4,5},
	{4,5,6,7,8,9},
	{5,6,7,9,10,11},
	{9,10,11,13,14,15}
};

// const std::vector< std::vector<int> > feats = {
// 	{0,1,2,3},
// 	{4,5,6,7}
// };

// const std::vector< std::vector<int> > feats = {
// 	{6,7,9,10,11,-1},
// 	{10,11,13,14,15,-1}
// 	// ,{0,1,2,4,5,8},
// 	// {4,5,6,8,9,10}
// };

typedef std::pair<board,board::reward> target;

#include <stack>
#include <iostream>

class ntuple : public weight_agent{
public:
	ntuple(const std::string& args = "") : weight_agent(args),op_code({0,1,2,3}){
		features.clear();
		for(auto vec : feats){
			features.push_back(IsoFeature(vec));
		}

		feature_num = 0;
		for(auto f : features){
			feature_num += f.get_all_feature().size();
		}
		//std::cout << feature_num;
	}

	virtual double get_value(const board& board) const{
		double sum = 0.0;
		for(int i=0;i<features.size();i++){
			for(const Feature& f : features[i].get_all_feature()){
				auto idx = f(board);
				sum += net[i][idx];
			}
		}

		// for(int i=0;i<16;i++){
		// 	if(board(i) >= 8){
		// 		sum += 500;
		// 		break;
		// 	}
		// }

		return sum;
	}

	virtual action take_action(const board& before){
		double best = -1;
		int best_op = -1;
		board::reward rew;
		for(int op : op_code){
			board after(before);
			board::reward reward = after.slide(op);
			if(reward != -1){
				double estimate = get_value(after) + reward;
				if(estimate > best){
					best = estimate;
					best_op = op;
					rew = reward;
				}
			}
		}

		if(best_op != -1){
			history.push(std::make_pair(before,rew));
			return action::slide(best_op);
		}else{
			history.push(std::make_pair(before,0));
			return action();
		}
	}
	
	void update_weight(double target, const board& curr){
		double value = get_value(curr);
		double tg = alpha/(double)(feature_num) * (target - value);
		for(int i=0;i<features.size();i++){
			for(const Feature& f: features[i].get_all_feature()){
				long long idx = f(curr);

				//does the update affecting the later update matters?
				net[i][idx] += tg;
			}
		}
		return;
	}

	virtual void open_episode(std::string flag = ""){
		while(!history.empty()) history.pop();
	}

	virtual void close_episode(std::string flag = ""){
		target current = history.top();
		history.pop();

		update_weight(0,current.first);
		board prev = current.first;
		while(!history.empty()){
			current = history.top();
			history.pop();

			double tg = current.second + get_value(prev);
			update_weight(tg,current.first);
			prev = current.first;
		}

		return;
	}

	
private:
	std::vector< IsoFeature > features;
	int feature_num;

	std::vector< int > op_code;

	std::stack< target > history;
};




/**
 * default random environment, i.e., placer
 * place the hint tile and decide a new hint tile
 */
class random_placer : public random_agent {
public:
	random_placer(const std::string& args = "") : random_agent("name=place role=placer " + args) {
		spaces[0] = { 12, 13, 14, 15 };
		spaces[1] = { 0, 4, 8, 12 };
		spaces[2] = { 0, 1, 2, 3};
		spaces[3] = { 3, 7, 11, 15 };
		spaces[4] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
	}

	virtual action take_action(const board& after) {
		std::vector<int> space = spaces[after.last()];
		std::shuffle(space.begin(), space.end(), engine);
		for (int pos : space) {
			if (after(pos) != 0) continue;

			int bag[3], num = 0;
			for (board::cell t = 1; t <= 3; t++)
				for (size_t i = 0; i < after.bag(t); i++)
					bag[num++] = t;
			std::shuffle(bag, bag + num, engine);

			board::cell tile = after.hint() ? after.hint() : bag[--num];
			board::cell hint = bag[--num];

			return action::place(pos, tile, hint);
		}
		return action();
	}

private:
	std::vector<int> spaces[5];
};

/**
 * random player, i.e., slider
 * select a legal action randomly
 */
class random_slider : public random_agent {
public:
	random_slider(const std::string& args = "") : random_agent("name=slide role=slider " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		std::shuffle(opcode.begin(), opcode.end(), engine);
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if (reward != -1) return action::slide(op);
		}
		return action();
	}

private:
	std::array<int, 4> opcode;
};

/*
Simple greedy
Move toward the best reward action
*/

class greedy_slider : public random_agent{
public:
	greedy_slider(const std::string& args = "") : random_agent("name=greedy role=slider" + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before){
		board::reward maximum_reward = 0;
		int best_op = -1;
		for(int op : opcode){
			board::reward op_reward = board(before).slide(op);
			if(op_reward > maximum_reward){
				maximum_reward = op_reward;
				best_op = op;
			}
		}

		return action::slide(best_op);
	}

private:
	std::array<int, 4> opcode;
};

/*
Move-restricting Greedy
No up and right move unless obligtory.

avg 876
*/
class MR_greedy_slider : public random_agent{
public:
	MR_greedy_slider(const std::string& args = "") : random_agent("name=greedy role=slider" + args),
		opcode({ 1, 2 }) {}

	virtual action take_action(const board& before){
		board::reward maximum_reward = 0;
		int best_op = -1;
		for(int op : opcode){
			board::reward op_reward = board(before).slide(op);
			if(op_reward >= maximum_reward){
				maximum_reward = op_reward;
				best_op = op;
			}
		}

		if(best_op == -1){
			board::reward zero_rew = board(before).slide(0);
			board::reward three_rew = board(before).slide(3);

			if(zero_rew == -1 && three_rew == -1)
				return action();
			else if(zero_rew == -1)
				return action::slide(3);
			else if(three_rew == -1)
				return action::slide(0);
			else
				return zero_rew < three_rew ? action::slide(3) : action::slide(0);
		}

		//if (best_op == -1) return board(before).slide(0) == -1? action::slide(3) : action::slide(0);

		return action::slide(best_op);
	}

private:
	std::array<int, 2> opcode;
};

/*
just try if the agent is the opposite of greedy

very bad
*/
class ungreedy_slider : public random_agent{
public:
	ungreedy_slider(const std::string& args = "") : random_agent("name=greedy role=slider" + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before){
		board::reward minimum_reward = INT16_MAX;
		int best_op = -1;
		for(int op : opcode){
			board::reward op_reward = board(before).slide(op);
			if(op_reward < minimum_reward && op_reward >= 0){
				minimum_reward = op_reward;
				best_op = op;
			}
		}

		return action::slide(best_op);
	}

private:
	std::array<int, 4> opcode;
};


/*
alternating greedy and ungreedy

bad
*/

class alternating_greedy_slider : public random_agent{
public:
	alternating_greedy_slider(const std::string& args = "") : random_agent("name=greedy role=slider" + args),
		opcode({ 0, 1, 2, 3 }),alternating(0) {}

	virtual action take_action(const board& before){
		alternating = !alternating;
		if(alternating)
			return take_action_greedily(before);
		else
			return take_action_ungreedily(before);
	}

private:

	action take_action_greedily(const board& before){
		board::reward maximum_reward = 0;
		int best_op = -1;
		for(int op : opcode){
			board::reward op_reward = board(before).slide(op);
			if(op_reward > maximum_reward){
				maximum_reward = op_reward;
				best_op = op;
			}
		}

		return action::slide(best_op);
	}

	action take_action_ungreedily(const board& before){
		board::reward minimum_reward = INT16_MAX;
		int best_op = -1;
		for(int op : opcode){
			board::reward op_reward = board(before).slide(op);
			if(op_reward < minimum_reward && op_reward >= 0){
				minimum_reward = op_reward;
				best_op = op;
			}
		}
		
		return action::slide(best_op);
	}

	bool alternating;
	std::array<int, 4> opcode;
};
