from typing import Dict, List, Optional
import logging
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from .types import ProductivityTechniques

logger = logging.getLogger(__name__)

class ScheduleItem(BaseModel):
    """A single item in the schedule."""
    time: str = Field(description="Time in 12-hour format (e.g., '9:00 AM', '2:30 PM')")
    task: str = Field(description="Specific, actionable task in simple language")

class ScheduleOutput(BaseModel):
    """The complete schedule output."""
    schedule: List[ScheduleItem] = Field(description="List of scheduled tasks")
    suggestions: List[str] = Field(description="List of suggestions and insights")
    techniques: str = Field(description="Brief explanation of how the day is structured")
    goal_alignment: Optional[str] = Field(description="How this schedule aligns with user's goals")
    missing_info: Optional[str] = Field(description="What additional information would help personalize the schedule")

class UserInput(BaseModel):
    """Input data for schedule generation."""
    goals: str = Field(description="Long-term goals")
    short_goals: str = Field(description="Short-term goals")
    bio: str = Field(description="User bio")
    tasks: str = Field(description="Today's tasks")
    patterns: List[Dict] = Field(description="Previous focus patterns")
    start_time: str = Field(description="Day start time")
    end_time: str = Field(description="Day end time")
    techniques: str = Field(description="Productivity techniques")

class ScheduleMemory:
    """Manages memory for schedule generation and updates."""
    def __init__(self, session_id: str):
        self.message_history = RedisChatMessageHistory(
            session_id=session_id,
            url="redis://localhost:6379"
        )
        self.memory = ConversationBufferMemory(
            memory_key="schedule_history",
            chat_memory=self.message_history,
            return_messages=True
        )

def get_schedule_prompt() -> ChatPromptTemplate:
    """Get the chat prompt template for schedule generation."""
    system_template = """
        You are a productivity AI assistant that creates personalized daily schedules. Your task is to make schedules that are:
        1. Easy to understand - avoid jargon and technical terms
        2. Specific and actionable - clear instructions that anyone can follow
        3. Personalized to the user's goals and patterns
        4. Structured with simple, effective techniques

        Use chain of thought reasoning to create the schedule:
        1. First, analyze the user's profile and goals
        2. Then, identify patterns and preferences from their history. 
            Find out user's MOST PRODUCTIVE TIME WINDOWS OF THE DAY
            ( for example, if user focus in afternoon for several days or 
            there is a consistent pattern of focus in the morning then 
            suggest the user to work on the most important task 
            in that time window & organise the schedule accordingly )
            also, EXPLAIN THE IDENTIFIED PRODUCTIVE WINDOWS in `techniques` field
            IDENTIFY USER'S CARDIAC RYTHEM FROM PATTERNS & 
            INCLUDE THIS INFORMATION IN `techniques` field
        3. Next, break down today's tasks and prioritize them
        4. Consider the time constraints and required breaks
        5. Finally, structure the day using appropriate techniques provided by user.
        6. CRUCIAL: RESTRUCTURE THE TASKS again by applying EISENHOWER MATRIX 
            INCLUDE THIS INFORMATION IN `techniques` field about how you sort the tasks 

        Response must be a valid JSON with:
        1. "schedule": list of tasks where each task has:
           - "time": in 24-hour format (e.g., "09:00", "22:30")
           - "task": specific, actionable task in simple language
        2. "suggestions": list of practical advice for the day
        3. "techniques": brief explanation of how the day is structured
        4. "goal_alignment": how this schedule helps with user's goals (if goals are provided)
        5. "missing_info": what additional information would help personalize the schedule better

        Include your reasoning in the "techniques" field to show how you arrived at this schedule.
        """

    human_template = """Let's think about this step by step:

                    1. First, let's understand the user:
                    - Long-term goals: {goals}
                    - Short-term goals: {short_goals}
                    - Bio: {bio}

                    2. Consider today's context:
                    - Tasks: {tasks}
                    - Previous patterns: {patterns}
                    - Available time: {start_time} to {end_time} ( time include user's wakeup time to sleep time)
                    - Available techniques: {techniques}

                    3. Guidelines for schedule creation:
                    - Use simple, clear language
                    - Make tasks extremely specific
                    - Include regular breaks (15 mins every 1-2 hours)
                    - Format times in 24-hour format (e.g., "09:00", "22:30")
                    - If bio/goals are missing, suggest updating them

                    4. Examples of good tasks:
                    "Write the first draft of the blog post about time management (aim for 500 words)"
                    "Take a 15-minute break: walk around or stretch"
                    "Check and respond to important emails (set a 30-minute limit)"
                    "Work on the most important task: [specific task from user's input]"

                    5. Examples to avoid:
                    "Use Pomodoro technique"
                    "Do deep work"
                    "Plan your day"
                    "Check emails"

                    Think through each step and explain your reasoning in the techniques section.
                    """

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])


def get_update_prompt() -> ChatPromptTemplate:
    """Get the chat prompt template for schedule updates."""
    system_template = """
    You are a productivity AI assistant that helps users adjust their schedules. Your task is to:
    1. Review the original schedule and its reasoning
    2. Understand the requested changes
    3. Validate if the changes maintain the schedule's effectiveness
    4. Update the schedule while preserving the original structure and goals

    Consider:
    1. Time conflicts
    2. Task dependencies
    3. Energy levels throughout the day
    4. Original productivity techniques used

    Response must be a valid JSON matching the original schedule format.
    Include brief explanations of how the changes affect the schedule in the techniques field.
    """

    human_template = """
    Original Schedule Context:
    {original_schedule}

    User Profile:
    - Goals: {goals}
    - Short-term goals: {short_goals}
    - Bio: {bio}

    Requested Changes:
    {changes}

    Please validate and incorporate these changes while maintaining the schedule's effectiveness.
    Explain any potential impacts or adjustments needed in the techniques field.
    """

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])

def generate_schedule_ai_gpt(
    previous_schedule: List[Dict],
    user_profile: Dict[str, str],
    todays_tasks: List[str],
    *,
    techniques: Optional[List[str]] = None,
    day_start_time: str = "6:00 AM",
    day_end_time: str = "9:30 PM",
) -> Dict:
    """Generate an AI-powered schedule using GPT-4.

    Args:
        previous_schedule: List of previous schedule items
        user_profile: User profile information
        todays_tasks: List of tasks for today
        techniques: Optional list of productivity techniques
        day_start_time: Start time in 12-hour format (default: "9:00 AM")
        day_end_time: End time in 12-hour format (default: "5:00 PM")

    Returns:
        Dict containing schedule and suggestions
    """
    try:
        # Initialize models with higher temperature for more creative reasoning
        llm = ChatOpenAI(model="o3-mini")
        prompt = get_schedule_prompt()
        parser = JsonOutputParser(pydantic_object=ScheduleOutput)

        # Prepare input data
        input_data = UserInput(
            goals=user_profile.get("long_term_goals", "Not provided"),
            short_goals=user_profile.get("short_term_goals", "Not provided"),
            bio=user_profile.get("bio", "Not provided"),
            tasks="\n".join(todays_tasks) if todays_tasks else "No specific tasks provided",
            patterns=previous_schedule,
            start_time=day_start_time,
            end_time=day_end_time,
            techniques=", ".join(techniques) if techniques else ProductivityTechniques.get_str()
        )

        # Generate and parse response
        messages = prompt.format_messages(**input_data.model_dump())
        response = llm.invoke(messages)

        logger.info("Total tokens in prompt: %d", sum(len(m.content) for m in messages))
        logger.info("Total tokens in response: %d", len(response.content))
        logger.info("Generating schedule with GPT-4... Techniques: %s", techniques)

        try:
            schedule_data = parser.parse(response.content)
            logger.debug("Schedule Data: %s", schedule_data)
            return schedule_data
        except Exception as e:
            from pprint import pprint
            pprint(response.content)
            logger.error("Error parsing AI response: %s", str(e))
            return {
                "schedule": [],
                "suggestions": ["Error generating schedule. Please try again."],
                "techniques": "Error occurred",
                "goal_alignment": None,
                "missing_info": None
            }
    except Exception as e:
        logger.error("Error in schedule generation: %s", str(e))
        return {
            "schedule": [],
            "suggestions": ["Error generating schedule. Please try again."],
            "techniques": "Error occurred",
            "goal_alignment": None,
            "missing_info": None
        }


def update_schedule_with_changes(
    original_schedule: Dict,
    updated_items: List[Dict],
    user_profile: Dict[str, str],
    session_id: Optional[str] = None
) -> Dict:
    """Update an existing schedule with user changes.

    Args:
        original_schedule: The original schedule with all metadata
        updated_items: List of updated schedule items
        user_profile: User profile information
        session_id: Optional session ID for memory persistence

    Returns:
        Updated schedule data
    """
    try:
        # Initialize memory if session_id provided
        memory = None
        if session_id:
            memory = ScheduleMemory(session_id)

        # Initialize model with lower temperature for more consistent updates
        llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        prompt = get_update_prompt()
        parser = JsonOutputParser(pydantic_object=ScheduleOutput)

        # Format the changes for the prompt
        changes_description = "\n".join([
            f"- Change task at {item['time']} to: {item['task']}"
            for item in updated_items
        ])

        # Prepare input data
        input_data = {
            "original_schedule": original_schedule,
            "goals": user_profile.get("goals", "Not provided"),
            "short_goals": user_profile.get("short_goals", "Not provided"),
            "bio": user_profile.get("bio", "Not provided"),
            "changes": changes_description
        }

        # Add memory context if available
        if memory:
            input_data["history"] = memory.memory.buffer

        # Generate and parse response
        messages = prompt.format_messages(**input_data)
        response = llm.invoke(messages)

        # Store interaction in memory if available
        if memory:
            memory.memory.save_context(
                {"input": changes_description},
                {"output": response.content}
            )

        try:
            updated_data = parser.parse(response.content)
            logger.debug("Updated Schedule Data: %s", updated_data)
            return updated_data
        except Exception as e:
            logger.error("Error parsing AI response for schedule update: %s", str(e))
            return original_schedule

    except Exception as e:
        logger.error("Error in schedule update: %s", str(e))
        return original_schedule


def fetch_user_todays_thoughts_and_tasks_dump_from_post_requests(request) -> List[str]:
    """Extract tasks and thoughts from POST request data."""
    tasks = []
    for key, value in request.POST.items():
        if key.startswith('task-') and value.strip():
            tasks.append(value.strip())
    return tasks
