document.addEventListener('DOMContentLoaded', function() {
    // 初始化可视化
    const svg = d3.select("#rl-visualization")
        .append("svg")
        .attr("width", "100%")
        .attr("height", "100%");
    
    // 概念数据
    const concepts = {
        overview: {
            title: "强化学习概述",
            description: "强化学习是机器学习的一个分支，它关注智能体(agent)如何在环境(environment)中采取行动(action)以获得最大的累积奖励(reward)。核心概念包括状态(state)、策略(policy)和价值函数(value function)。",
            animation: drawOverview
        },
        agent: {
            title: "智能体 (Agent)",
            description: "智能体是强化学习中的学习者和决策者。它通过观察环境状态，选择行动，并从环境中接收奖励来学习。智能体的目标是学习一个策略，使得长期累积奖励最大化。",
            animation: drawAgent
        },
        environment: {
            title: "环境 (Environment)",
            description: "环境是智能体交互的外部系统。它接收智能体的行动，返回新的状态和奖励。环境可以是完全或部分可观测的，决定了智能体能看到多少信息。",
            // animation: drawEnvironment
        },
        state: {
            title: "状态 (State)",
            description: "状态是环境在特定时间点的完整描述。在完全可观测环境中，状态等同于智能体的观察；在部分可观测环境中，状态可能包含隐藏信息。",
            // animation: drawState
        },
        action: {
            title: "行动 (Action)",
            description: "行动是智能体在特定状态下可以执行的操作。行动空间可以是离散的(如上下左右)或连续的(如速度、角度)。",
            // animation: drawAction
        },
        reward: {
            title: "奖励 (Reward)",
            description: "奖励是环境对智能体行动的即时反馈信号。强化学习的目标是最大化长期累积奖励，而不仅仅是即时奖励。",
            // animation: drawReward
        },
        policy: {
            title: "策略 (Policy)",
            description: "策略是智能体的行为函数，定义了在给定状态下选择行动的方式。策略可以是确定性的(明确选择某个行动)或随机性的(按概率分布选择行动)。",
            // animation: drawPolicy
        },
        value: {
            title: "价值函数 (Value Function)",
            description: "价值函数评估状态或状态-行动对的长期价值，表示从该状态开始，遵循特定策略能获得的预期累积奖励。",
            // animation: drawValueFunction
        }
    };
    
    // 按钮点击事件
    document.querySelectorAll('.concept-btn').forEach(button => {
        button.addEventListener('click', function() {
            const concept = this.getAttribute('data-concept');
            
            // 更新活动按钮
            document.querySelectorAll('.concept-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            this.classList.add('active');
            
            // 更新内容
            document.getElementById('concept-title').textContent = concepts[concept].title;
            document.getElementById('concept-description').innerHTML = `<p>${concepts[concept].description}</p>`;
            
            // 清除旧动画
            d3.select("#rl-visualization svg").selectAll("*").remove();
            
            // 绘制新动画
            concepts[concept].animation();
        });
    });
    
    // 初始绘制概述
    drawOverview();
    
    // 绘图函数
    function drawOverview() {
        const svg = d3.select("#rl-visualization svg");
        const width = document.getElementById('rl-visualization').clientWidth;
        const height = document.getElementById('rl-visualization').clientHeight;
        
        // 绘制智能体
        const agent = svg.append("circle")
            .attr("class", "agent")
            .attr("cx", width * 0.3)
            .attr("cy", height * 0.5)
            .attr("r", 30)
            .attr("fill", "#e74c3c");
        
        // 绘制环境
        const environment = svg.append("rect")
            .attr("class", "environment")
            .attr("x", width * 0.5)
            .attr("y", height * 0.2)
            .attr("width", width * 0.4)
            .attr("height", height * 0.6)
            .attr("rx", 10)
            .attr("fill", "#2ecc71")
            .attr("opacity", 0.2)
            .attr("stroke", "#2ecc71")
            .attr("stroke-width", 2);
        
        // 绘制状态
        const states = [];
        for (let i = 0; i < 3; i++) {
            states.push(svg.append("circle")
                .attr("class", "state")
                .attr("cx", width * (0.55 + i * 0.15))
                .attr("cy", height * (0.3 + i * 0.2))
                .attr("r", 20)
                .attr("fill", "#3498db"));
        }
        
        // 绘制行动箭头
        const actions = [];
        for (let i = 0; i < 2; i++) {
            actions.push(svg.append("path")
                .attr("class", "action")
                .attr("d", `M ${width * 0.3} ${height * 0.5} L ${width * 0.5} ${height * (0.3 + i * 0.4)}`)
                .attr("fill", "none")
                .attr("marker-end", "url(#arrowhead)"));
        }
        
        // 绘制奖励
        const rewards = [];
        for (let i = 0; i < 2; i++) {
            rewards.push(svg.append("path")
                .attr("class", "reward")
                .attr("d", `M ${width * (0.55 + i * 0.15)} ${height * (0.3 + i * 0.2)} L ${width * 0.3} ${height * 0.7}`)
                .attr("fill", "none")
                .attr("stroke", "#f1c40f")
                .attr("stroke-width", 2)
                .attr("marker-end", "url(#rewardhead)"));
        }
        
        // 添加箭头标记
        svg.append("defs").append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 8)
            .attr("refY", 0)
            .attr("orient", "auto")
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("xoverflow", "visible")
            .append("path")
            .attr("d", "M 0,-5 L 10,0 L 0,5")
            .attr("fill", "#f39c12");
            
        svg.append("defs").append("marker")
            .attr("id", "rewardhead")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 8)
            .attr("refY", 0)
            .attr("orient", "auto")
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("xoverflow", "visible")
            .append("path")
            .attr("d", "M 0,-5 L 10,0 L 0,5")
            .attr("fill", "#f1c40f");
        
        // 添加标签
        svg.append("text")
            .attr("x", width * 0.3)
            .attr("y", height * 0.5 - 40)
            .attr("text-anchor", "middle")
            .text("Agent");
            
        svg.append("text")
            .attr("x", width * 0.7)
            .attr("y", height * 0.1)
            .attr("text-anchor", "middle")
            .text("Environment");
            
        states.forEach((state, i) => {
            svg.append("text")
                .attr("x", width * (0.55 + i * 0.15))
                .attr("y", height * (0.3 + i * 0.2) - 30)
                .attr("text-anchor", "middle")
                .text(`State ${i+1}`);
        });
        
        // 添加动画
        animateOverview();
    }
    
    function animateOverview() {
        // 实现概述动画
        const svg = d3.select("#rl-visualization svg");
        
        // 智能体脉冲动画
        d3.select(".agent")
            .transition()
            .duration(1000)
            .attr("r", 35)
            .transition()
            .duration(1000)
            .attr("r", 30)
            .on("end", function repeat() {
                d3.select(this)
                    .transition()
                    .duration(1000)
                    .attr("r", 35)
                    .transition()
                    .duration(1000)
                    .attr("r", 30)
                    .on("end", repeat);
            });
        
        // 行动箭头动画
        d3.selectAll(".action")
            .attr("stroke-dasharray", "10,5")
            .attr("stroke-dashoffset", 0)
            .transition()
            .duration(2000)
            .attr("stroke-dashoffset", -15)
            .on("end", function repeat() {
                d3.select(this)
                    .attr("stroke-dashoffset", 0)
                    .transition()
                    .duration(2000)
                    .attr("stroke-dashoffset", -15)
                    .on("end", repeat);
            });
        
        // 奖励箭头动画
        d3.selectAll(".reward")
            .attr("stroke-dasharray", "10,5")
            .attr("stroke-dashoffset", 0)
            .transition()
            .delay(1000)
            .duration(2000)
            .attr("stroke-dashoffset", -15)
            .on("end", function repeat() {
                d3.select(this)
                    .attr("stroke-dashoffset", 0)
                    .transition()
                    .delay(1000)
                    .duration(2000)
                    .attr("stroke-dashoffset", -15)
                    .on("end", repeat);
            });
    }
    
    function drawAgent() {
        // 实现Agent的可视化
        const svg = d3.select("#rl-visualization svg");
        const width = document.getElementById('rl-visualization').clientWidth;
        const height = document.getElementById('rl-visualization').clientHeight;
        
        // 绘制智能体
        const agent = svg.append("circle")
            .attr("class", "agent")
            .attr("cx", width * 0.5)
            .attr("cy", height * 0.5)
            .attr("r", 50)
            .attr("fill", "#e74c3c");
        
        // 添加眼睛
        svg.append("circle")
            .attr("cx", width * 0.48)
            .attr("cy", height * 0.48)
            .attr("r", 8)
            .attr("fill", "white");
            
        svg.append("circle")
            .attr("cx", width * 0.52)
            .attr("cy", height * 0.48)
            .attr("r", 8)
            .attr("fill", "white");
            
        // 添加嘴巴
        svg.append("path")
            .attr("d", `M ${width * 0.45} ${height * 0.55} Q ${width * 0.5} ${height * 0.6}, ${width * 0.55} ${height * 0.55}`)
            .attr("fill", "none")
            .attr("stroke", "white")
            .attr("stroke-width", 3);
        
        // 添加标签
        svg.append("text")
            .attr("x", width * 0.5)
            .attr("y", height * 0.5 - 70)
            .attr("text-anchor", "middle")
            .attr("font-size", "24px")
            .text("智能体 (Agent)");
            
        svg.append("text")
            .attr("x", width * 0.5)
            .attr("y", height * 0.5 + 90)
            .attr("text-anchor", "middle")
            .attr("font-size", "16px")
            .text("观察环境 → 采取行动 → 接收奖励 → 学习改进");
        
        // 添加动画
        animateAgent();
    }
    
    function animateAgent() {
        // 实现Agent动画
        const svg = d3.select("#rl-visualization svg");
        const width = document.getElementById('rl-visualization').clientWidth;
        const height = document.getElementById('rl-visualization').clientHeight;
        
        // 观察动画
        const eyeLeft = svg.append("circle")
            .attr("cx", width * 0.48)
            .attr("cy", height * 0.48)
            .attr("r", 4)
            .attr("fill", "black");
            
        const eyeRight = svg.append("circle")
            .attr("cx", width * 0.52)
            .attr("cy", height * 0.48)
            .attr("r", 4)
            .attr("fill", "black");
            
        function observe() {
            eyeLeft.transition()
                .duration(500)
                .attr("cx", width * 0.45)
                .transition()
                .duration(500)
                .attr("cx", width * 0.48);
                
            eyeRight.transition()
                .duration(500)
                .attr("cx", width * 0.55)
                .transition()
                .duration(500)
                .attr("cx", width * 0.52);
        }
        
        // 嘴巴动画
        const mouth = d3.select("path[d^='M']");
        
        function act() {
            mouth.transition()
                .duration(300)
                .attr("d", `M ${width * 0.45} ${height * 0.55} Q ${width * 0.5} ${height * 0.65}, ${width * 0.55} ${height * 0.55}`)
                .transition()
                .duration(300)
                .attr("d", `M ${width * 0.45} ${height * 0.55} Q ${width * 0.5} ${height * 0.6}, ${width * 0.55} ${height * 0.55}`);
        }
        
        // 整体动画循环
        function agentLoop() {
            observe();
            setTimeout(act, 1000);
            setTimeout(agentLoop, 2000);
        }
        
        agentLoop();
    }
    
    // 其他绘图函数 (drawEnvironment, drawState, 等) 类似实现
    // 由于篇幅限制，这里只展示了部分实现
    
    // function drawEnvironment() {
    //     // 实现Environment的可视化
    //     // 类似上面的实现方式
    // }
    
    // function drawState() {
    //     // 实现State的可视化
    //     // 类似上面的实现方式
    // }
    
    // 其他概念的可视化函数...
});